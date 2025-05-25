import sglang as sgl
import datasets
import argparse
import os
import sys
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--prompt_mode", type=str, default="full_template")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n", type=int, default=32)
args = parser.parse_args()

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

def apply_simple_steer_template(x):
    return f"Question: {x[0]['content']}\nAnswer: \n\n"

def apply_minimum_dialog_template(x):
    text = ""
    for message in x:
        text += f"{message['role']}: {message['content']}\n"
    text += "assistant: "
    return text.rstrip()

def get_prompts(item):
    message = [
        {"role": "user", "content": item['instruction']},
    ]
    if "llama" in args.model.lower():
        if args.prompt_mode == "full_template":
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        elif args.prompt_mode == "simple_steer":
            prompt = apply_simple_steer_template(message)
        elif args.prompt_mode == "minimum_dialog":
            prompt = apply_minimum_dialog_template(message)
    return prompt

def generate_responses(llm, sampling_params, batched_prompts, batched_original_prompts):
    responses = llm.generate(batched_prompts, sampling_params)
    # split responses by n
    batched_responses = []
    for i in range(0, len(responses), args.n):
        original_prompt = batched_original_prompts[i // args.n]
        batch = {
            'prompt': original_prompt,
            'prompt_w_template': batched_prompts[i // args.n],
            'responses': []
        }
        for j in range(i, i+args.n):
            batch['responses'].append(responses[j]['text'])
        batched_responses.append(batch)
    return batched_responses

def get_reward_score(prompt, response):
    # tokenize the prompt and responses
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    tokenized_conv = rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device)
    with torch.no_grad():
        score = rm(tokenized_conv).logits[0][0].item()
    return score

if __name__ == "__main__":
    sampling_params = {
        "temperature": args.temperature, 
        "top_p": args.top_p, 
        "repetition_penalty": args.repetition_penalty, 
        "max_new_tokens": args.max_new_tokens,
        "n": args.n
    }
    prompts = []
    original_prompts = []
    for item in eval_set:
        prompts.append(get_prompts(item))
        original_prompts.append(item['instruction'])
    # split prompts into batches
    batched_prompts = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    batched_original_prompts = [original_prompts[i:i+args.batch_size] for i in range(0, len(original_prompts), args.batch_size)]
    llm = sgl.Engine(model_path=args.model, log_level="info")

    all_outputs = []
    for batched_prompt, batched_original_prompt in zip(batched_prompts, batched_original_prompts):
        batched_outputs = generate_responses(llm, sampling_params, batched_prompt, batched_original_prompt)
        all_outputs.extend(batched_outputs)
    llm.shutdown()

    # use reward model to select the best output and construct the dataset

    # Load model and tokenizer
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get the reward score for each response
    for output in tqdm(all_outputs):
        prompt = output['prompt']
        responses = output['responses']
        scores = []
        for response in responses:
            score = get_reward_score(prompt, response)
            scores.append(score)
        output['scores'] = scores
    
    # save the outputs
    output_name = f"alpaca_eval/data/alpaca_eval_outputs_{args.model.split('/')[-1]}_{args.prompt_mode}.json"
    with open(output_name, "w") as f:
        json.dump(all_outputs, f)
