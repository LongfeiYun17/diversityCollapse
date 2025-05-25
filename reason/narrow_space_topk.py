import os
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import entropy
import json
import pickle as pkl
from pathlib import Path
from torch.nn import functional as F
class LLMDiverseHeadTester:
    def __init__(self, layer_num, head_num, model_name, model_id, question, total_steps=10, mode="full_template"):
        self.layer_num = layer_num
        self.head_num = head_num
        self.model_name = model_name
        self.model_id = model_id
        self.model_to_test = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16, device_map='auto').eval()
        self.question = question
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.chat_template is None:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.total_steps = total_steps
        self.mode = mode

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, return_tensors='pt'):
        text = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=tokenize, 
            add_generation_prompt=add_generation_prompt, 
            return_tensors=return_tensors
        )
        return text

    def evaluate_and_log(self):
        prompt = [
            {"role": "user", "content": f"{self.question}"},
        ]
        if self.mode == "full_template":
            input_ids = self.apply_chat_template(conversation=prompt, tokenize=True,  add_generation_prompt=True, return_tensors='pt').to(self.model_to_test.device)
        elif self.mode == "simple_steer":
            input_ids = self.tokenizer.encode(self.question, add_special_tokens=False, return_tensors='pt').to(self.model_to_test.device)
        question = self.tokenizer.decode(input_ids[0, :])
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            output, entropies = self.decode(q_outputs, input_ids[:,-1], self.total_steps)
        answer = self.tokenizer.decode(output)
        print(f'mode: {self.mode}, question: {question}, answer: {answer}, ')
        return output, entropies 

    def decode(self, q_outputs, inp, decode_len, block_list=None):
        output, entropies = [], []

        past_key_values = q_outputs.past_key_values
        last_token = inp.view(1, 1)

        for step in range(decode_len):
            outputs = self.model_to_test(
                input_ids=last_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )
            logits = outputs.logits[0, -1]
            # Get top 10 logits and compute probabilities just for those
            top_k_logits, top_k_indices = torch.topk(logits, k=10)
            top_k_probs = F.softmax(top_k_logits, dim=-1).cpu().numpy()
            token_id = top_k_indices[0].item()  # Take highest probability token
            
            output.append(token_id)
            entropies.append(entropy(top_k_probs))

            last_token = torch.tensor([[token_id]], device=self.model_to_test.device)
            past_key_values = outputs.past_key_values
            #print(f'step: {step}, length of key values: {past_key_values[0][0].shape[-2]}')

        return output, entropies

    def diversity_calculate(self, attention_maxtrix, diversity_score, inp, step_token, topk=1):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                for v, i in zip(values, idx):
                    if i < len(self.prompt_ids):
                        idx = i.item()
                        #token_id = self.prompt_ids[i].item()
                        if not (idx >= self.question_start_pos and idx < self.question_end_pos):
                            diversity_score[layer_idx][head_idx][0] += 1 / self.total_steps
                            diversity_score[layer_idx][head_idx][1] += step_token
                        break
        
def autopct_func(pct):
    return f"{pct:.1f}%" if pct >= 1 else ""
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default="llama", help='id of model')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='name of model')
    parser.add_argument('--total_steps', type=int, default=10, help='total steps')
    parser.add_argument('--question', type=str, default="Please write a news about a random topic.", help='question')
    parser.add_argument('--question_id', type=str, default="01", help='question id')
    args = parser.parse_args()
    if Path(f'reason/entropies_topk/{args.id}/{args.question_id}.pkl').exists():
        print(f'reason/entropies_topk/{args.id}/{args.question_id}.pkl already exists')
        exit()

    llm_diverse_head_tester = LLMDiverseHeadTester(
        layer_num=28, 
        head_num=24, 
        model_name=args.model_name, 
        model_id=args.id, 
        question=args.question, 
        total_steps=args.total_steps,
        mode="full_template"
    )

    _, full_template_entropies = llm_diverse_head_tester.evaluate_and_log()

    llm_diverse_head_tester = LLMDiverseHeadTester(
        layer_num=28, 
        head_num=24, 
        model_name=args.model_name, 
        model_id=args.id, 
        question=args.question, 
        total_steps=args.total_steps,
        mode="simple_steer"
    )

    _, simple_steer_entropies = llm_diverse_head_tester.evaluate_and_log()  

    entropy_dir = f'reason/entropies_topk/{args.id}'
    if not os.path.exists(entropy_dir):
        os.makedirs(entropy_dir)
    with open(f'{entropy_dir}/{args.question_id}.pkl', 'wb') as f:
        pkl.dump({'full_template': full_template_entropies, 'simple_steer': simple_steer_entropies}, f)
    # plot the entropies
    plt.plot(full_template_entropies, label='full_template')
    plt.plot(simple_steer_entropies, label='simple_steer')
    plt.legend()
    output_dir = f'reason/figs_topk/{args.id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/{args.question_id}.png')