import wandb
import os
import torch 
import codecs
import json  
import transformers 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer 

class PhiTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )

        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            tax_lines = f.readlines()
        tax_pairs = [line.strip().split("\t") for line in tax_lines]

        self.tax_graph = TaxStruct(tax_pairs) 
        self.sampler = Sampler(self.tax_graph) 
        
        # if 'TaxStruct' not in globals():
        #     print("Warning: 'TaxStruct' class not found. Using dummy structure.")
        #     self.tax_graph = None
        # else:
        #     self.tax_graph = TaxStruct(tax_pairs)

        # if 'Sampler' not in globals():
        #     print("Warning: 'Sampler' class not found. Using dummy structure.")
        #     self.sampler = None
        # else:
        #      self.sampler = Sampler(self.tax_graph)

        self._tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_path,
            trust_remote_code=True,
            cache_dir="./cache/"
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self._tokenizer.pad_token}")

        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            self._word2des = json.load(fp)
            
        # if 'PhiTaxonomyModel' not in globals():
        #     print("Error: 'PhiTaxonomyModel' class not found. This code will fail.")
        #     self.model = None # This will cause an error later, but shows dependency
        # else:
        self.model = PhiTaxonomyModel(
            args.pretrained_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

    def train(self):

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay
        )
        dataset = PhiDataset(
            self.sampler,
            tokenizer=self._tokenizer,
            word2des=self._word2des,
            padding_max=self.args.padding_max,
            margin_beta=self.args.margin_beta
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        total_steps = len(data_loader) * self.args.epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )

        self.model.train()
        global_step = 0
        best_loss = float('inf')
        patience_counter = 0

        print(f"Starting training for {self.args.epochs} epochs")
        print(f"Total steps: {total_steps}, Warmup steps: {self.args.warmup_steps}")

        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")

            dataset.resample()
            data_loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
                try:
                    optimizer.zero_grad()
                    pos_ids = batch["pos_ids"].to(self.device)
                    pos_attn = batch["pos_attn_masks"].to(self.device)
                    neg_ids = batch["neg_ids"].to(self.device)
                    neg_attn = batch["neg_attn_masks"].to(self.device)
                    margin = batch["margin"].to(self.device)

                    pos_output = self.model(input_ids=pos_ids, attention_mask=pos_attn)
                    neg_output = self.model(input_ids=neg_ids, attention_mask=neg_attn)

                    if torch.isnan(pos_output).any() or torch.isnan(neg_output).any():
                        print(f"\nNaN detected in outputs at step {global_step}!")
                        print("Stopping training to prevent gradient corruption.")
                        return False

                    if torch.isinf(pos_output).any() or torch.isinf(neg_output).any():
                        print(f"\nInf detected in outputs at step {global_step}!")
                        print("Stopping training.")
                        return False

                    loss = self.model.margin_loss_fct(pos_output, neg_output, margin)

                    if torch.isnan(loss):
                        print(f"\n NaN loss at step {global_step}!")
                        return False

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    if global_step % 10 == 0:
                        wandb.log({
                            "loss": loss.item(),
                            "epoch": epoch,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "pos_score_mean": pos_output.mean().item(),
                            "neg_score_mean": neg_output.mean().item(),
                            "score_diff": (pos_output.mean() - neg_output.mean()).item()
                        }, step=global_step)

                    global_step += 1

                    if global_step % self.args.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{global_step}")
                        print(f"\nCheckpoint saved at step {global_step}")

                except RuntimeError as e:
                    print(f"Runtime error at step {global_step}: {e}")
                    if "out of memory" in str(e):
                        print("OOM error")
                        torch.cuda.empty_cache()
                    return False

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            print(f"Epoch {epoch + 1}:")
            print(f"Average Loss: {avg_epoch_loss:.6f}")
            print(f"Best Loss So Far: {best_loss:.6f}")

            epoch_checkpoint_name = f"epoch-{epoch + 1}"
            self.save_checkpoint(epoch_checkpoint_name)
            print(f"Epoch {epoch + 1} checkpoint saved to {self.args.save_path}/{epoch_checkpoint_name}")

            if avg_epoch_loss < best_loss:
                improvement = best_loss - avg_epoch_loss
                best_loss = avg_epoch_loss
                patience_counter = 0
                self.save_model()
                print(f"New best model saved (improved by {improvement:.6f})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.args.patience}")

            if patience_counter >= self.args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.args.patience} consecutive epochs")
                break

            if epoch > 2 and avg_epoch_loss > best_loss * 2:
                print(f"\nTraining diverging? (loss: {avg_epoch_loss:.4f} vs best: {best_loss:.4f})")
                print("Reduce lr? or check again")
                break

        print(f"Best loss achieved: {best_loss:.6f}")
        return True

    def save_model(self):
        os.makedirs(self.args.save_path, exist_ok=True)

        self.model.phi_model.save_pretrained(self.args.save_path)
        self._tokenizer.save_pretrained(self.args.save_path)

        torch.save(
            self.model.classifier.state_dict(),
            f"{self.args.save_path}/classifier.pt"
        )

        print(f"Model saved to {self.args.save_path}")

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = f"{self.args.save_path}/{checkpoint_name}"
        os.makedirs(checkpoint_path, exist_ok=True)

        self.model.phi_model.save_pretrained(checkpoint_path)
        torch.save(
            self.model.classifier.state_dict(),
            f"{checkpoint_path}/classifier.pt"
        )


class TrainingArgs:
    def __init__(self):
        self.taxo_path = "science_train.taxo"
        # self.val_taxo_path = '/content/science_val_combined.taxo'
        self.dic_path = "dic.json"
        self.pretrained_path = "Qwen/Qwen2-1.5B-Instruct"
        self.save_path = "./output/qwen_instruct_taxonomy"

        self.epochs = 10
        self.batch_size = 8
        # self.lr = 2e-4
        self.lr = 5e-5
        self.eps = 1e-8
        self.weight_decay = 0.01
        self.warmup_steps = 5
        self.save_steps = 50
        self.max_grad_norm = 1.0
        self.patience = 3

        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1

        self.padding_max = 512
        self.margin_beta = 0.2

        self.wandb_project = "taxonomy-placement"
        self.experiment_name = "qwen-instruct-lora-1.5"

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = TrainingArgs()
    trainer = PhiTrainer(args)
    success = trainer.train()

    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed or stopped early.")
