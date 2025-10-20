import torch
import codecs
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

class PhiEval:
    def __init__(self, args):
        self.args = args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            tax_lines = f.readlines()

        tax_pairs = []
        for line in tax_lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                _, child, parent = parts
                tax_pairs.append((parent.strip(), child.strip()))
            elif len(parts) == 2:
                child, parent = parts
                tax_pairs.append((child.strip(), parent.strip()))
            else:
                print(f"Skipping malformed line: {line.strip()}")

        self.tax_graph = TaxStruct(tax_pairs)
        self.sampler = Sampler(self.tax_graph)

        with codecs.open(args.terms, 'r', encoding='utf-8') as f:
            self.terms = [line.strip() for line in f.readlines()]

        self.tokenizer = AutoTokenizer.from_pretrained('/content/content/output/qwen_instruct_taxonomy')

        # self.model = PhiTaxonomyModel(args.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path)
        self.model.to(self.device)

        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        # self.classifier = nn.Linear(hidden_size, 1)
        classifier_path = f"/content/content/output/qwen_instruct_taxonomy/epoch-2/classifier.pt"
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.to(self.device)
        self.model.eval()
        self.classifier.eval()

        self.dropout = nn.Dropout(0.1)
        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            word2des = json.load(fp)

        self._dataset = PhiDataset(None, self.tokenizer, word2des, args.padding_max, margin_beta=0.2)

    def gen_eval_batches(self, term, batch_size=4):
        paths = list(self.tax_graph.node2path.values())
        num_paths = len(paths)

        for i in range(0, num_paths, batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_ids, batch_attn = [], []

            for path in batch_paths:
                ids, attn_mask = self._dataset.encode_path_as_prompt([term] + path)
                batch_ids.append(ids)
                batch_attn.append(attn_mask)

            yield torch.stack(batch_ids, dim=0).to(self.device), torch.stack(batch_attn, dim=0).to(self.device)

    def predict(self, batch_size=4):
        tags = list(self.tax_graph.node2path.keys())
        results = []

        # terms_to_evaluate = self.terms[]
        terms_to_evaluate = self.terms[40:]

        for term in tqdm(terms_to_evaluate, desc="Evaluating", total=len(terms_to_evaluate)):
            all_scores = []

            for input_ids, attn_mask in self.gen_eval_batches(term, batch_size=batch_size):

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True
                    )

                    hidden_states = outputs.hidden_states[-1] 
                    sequence_lengths = attn_mask.sum(dim=1) - 1
                    batch_size_seq = input_ids.shape[0]
                    pooled_output = hidden_states[
                        torch.arange(batch_size_seq, device=hidden_states.device), sequence_lengths
                    ]
                    pooled_output = pooled_output.to(self.classifier.weight.dtype)
                    # input_ids = input_ids.to(self.device)
                    # attn_mask = attn_mask.to(self.device)
                    pooled_output = pooled_output.to(self.device)  # just to be safe

                    # pooled_output = self.dropout(pooled_output)
                    with torch.no_grad():
                        logits = self.classifier(pooled_output).squeeze(-1)
                    # logits = self.classifier(pooled_output).squeeze(-1)

                all_scores.append(logits.cpu())

                del input_ids, attn_mask, outputs, hidden_states, pooled_output, logits
                torch.cuda.empty_cache()

            all_scores = torch.cat(all_scores, dim=0)

            _, indices = all_scores.sort(descending=True)
            ranked = [tags[int(i)] for i in indices]
            results.append(ranked)

        return results

    def save_results(self, results):

        with codecs.open(self.args.output, mode='w+', encoding='utf-8') as fp:
            fp.write("\n".join(["\t".join(r) for r in results]))
