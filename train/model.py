import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import codecs
import json


class PhiTaxonomyModel(nn.Module):
    EPS = 1e-9

    def __init__(self, pretrained_path, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.phi_model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto",
            output_hidden_states=True,
            cache_dir="./cache/"
        )

        self.phi_model = prepare_model_for_kbit_training(self.phi_model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj"],
            bias="none",
            inference_mode=False
        )

        self.phi_model = get_peft_model(self.phi_model, lora_config)
        self.phi_model.print_trainable_parameters()

        hidden_size = self.phi_model.config.hidden_size

        self.dropout = nn.Dropout(lora_dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        if torch.cuda.is_available():
            self.dropout = self.dropout.cuda()
            self.classifier = self.classifier.cuda()

        print(f"Hidden size: {hidden_size}")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        for name, param in self.phi_model.named_parameters():
          if param.requires_grad:
              print(f"{name} is trainable")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        Forward pass through Phi model

        Args:
            input_ids: tokenized input
            attention_mask: attention mask
            labels: optional labels for loss computation

        Returns:
            logits or loss
        """
        outputs = self.phi_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]

        # Using last non-padded token (equivalent to CLS/pooled output for causal LMs), no CLS token, initialling made mistake of taking last token irrespective of padding, so faced issues with training
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        pooled_output = pooled_output.to(self.classifier.weight.dtype)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            return loss_fct(logits.squeeze(), labels.squeeze())

        # return logits
        return logits.squeeze(-1)

    ''' used my own implementation of the loss in the previous iteration, very unstable and gradient explosion was too common, so ended up using the one used in TEMP's official implemenatation'''

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        # loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
        #         neg_score.squeeze().relu().clamp(min=cls.EPS) +
        #         margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        # return loss.sum()

        loss = torch.relu(neg_score.squeeze() - pos_score.squeeze() + margin)
        return loss.mean()
