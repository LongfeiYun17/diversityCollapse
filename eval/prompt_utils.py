import random
from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tulu_tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
all_tokenizers = [llama_tokenizer, tulu_tokenizer, mistral_tokenizer, qwen_tokenizer, phi_tokenizer]

def apply_simple_steer_template(x):
    return f"{x[0]['content']}"

def apply_chat_template(x, tokenizer):
    text = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    return text

def apply_natural_template(x):
    text = ""
    for message in x:
        text += f"{message['role']}: {message['content']}\n"
    text += "assistant: "
    return text.rstrip()

def apply_mixed_template(x):
    random_tokenizer = random.choice(all_tokenizers)
    text = random_tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
    return text


def get_prompts(task_name, dataset, args, content=None):
    prompts = []
    for prompt in dataset:
        if task_name == "common_gen":
            concepts = prompt['concepts']
            message = [
                {"role": "user", "content": f'Write a sentence that contains the following concepts: {concepts}.\n\n Sentence: '},
            ]
        elif task_name == "eli5":
            question = prompt['question']
            message = [
                {"role": "user", "content": f'Question: {question}\n\n Answer: '},
            ]
        elif task_name == "natural_questions":
            question = prompt['query']
            message = [
                {"role": "user", "content": f'Question: {question}\n\n Answer: '},
            ]
        elif task_name == "writingprompts":
            message = [
                {"role": "user", "content": f'Complete the following story: {prompt["prompt"]}'},
            ]
        elif task_name == "rocstory":
            message = [
                {"role": "user", "content": f'Complete the following story: {prompt["target"]}'},
            ]
        elif task_name == "story_cloze":
            message = [
                {"role": "user", "content": f'Complete the following story: {prompt["prompt"]}'},
            ]
        elif task_name == "news_generation":
            message = [
                {"role": "user", "content": f'Please write a {args.style} about a random topic.'},
            ]
        elif task_name == "travel_generation":
            message = [
                {"role": "user", "content": f"Please write a sentence about a city for travel. Sentence: "},
            ]
        elif task_name == "book_generation":
            message = [
                {"role": "user", "content": f"Please write a sentence to introduce a book you like. Sentence: "},
            ]
        else:
            message = [
                {"role": "user", "content": f"{content}"},
            ]

        if "llama" in args.mode.lower():
            if args.prompt_mode == "full_template":
                prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
            elif args.prompt_mode == "simple_steer":
                prompt = apply_simple_steer_template(message)
            elif args.prompt_mode == "minimum_dialog":
                prompt = f"""user\n{message[0]['content']}\nassistant\n"""
            elif args.prompt_mode == "fake_template":
                prompt = f"""<#init_text#><#random_header#>user<#/random_header#>\n\n{message[0]['content']}<#eod#><#random_header#>assistant<#/random_header#>\n\n"""
 
        elif "qwen3" in args.mode.lower():
            if args.prompt_mode == "full_template":
                prompt = f"""<|im_start|>user\n{message[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n""" 
            elif args.prompt_mode == "simple_steer":
                prompt = apply_simple_steer_template(message)
            elif args.prompt_mode == "minimum_dialog":
                prompt = f"""user\n{message[0]['content']}\nassistant\n"""
            elif args.prompt_mode == "fake_template":
                prompt = f"""<<#q_start#>>user\n{message[0]['content']}<<#q_end#>>\n<<#q_start#>>assistant\n<thought>\n\n</thought>\n\n"""

        elif "tulu" in args.mode.lower():
            if args.prompt_mode == "full_template": 
                prompt = f"""<|user|>\n{message[0]['content']}\n<|assistant|>\n\n"""
 
            elif args.prompt_mode == "minimum_dialog": 
                prompt = f"""user\n{message[0]['content']}\nassistant\n"""   
            elif args.prompt_mode == "simple_steer":
                prompt = apply_simple_steer_template(message)
            elif args.prompt_mode == "fake_template":
                prompt = f"<<@@user@@>>\n{message[0]['content']}\n<<@@assistant@@>>\n"

        elif "phi" in args.mode.lower():
            if args.prompt_mode == "full_template":
                prompt = f"""<|user|>\n{message[0]['content']}<|end|>\n<|assistant|>\n\n"""
            elif args.prompt_mode == "minimum_dialog":
                prompt = f"""user\n{message[0]['content']}\nassistant\n"""
            elif args.prompt_mode == "simple_steer":
                prompt = apply_simple_steer_template(message)
            elif args.prompt_mode == "fake_template":
                prompt = f"""<<#user#>> {message[0]['content']} <<#end#>>\n<<#assistant#>>"""
        else:
            if args.prompt_mode == "full_template":
                prompt = apply_chat_template(message)
            elif args.prompt_mode == "simple_steer":
                prompt = apply_simple_steer_template(message)
            elif args.prompt_mode == "mixed_template":
                prompt = apply_mixed_template(message)
        prompts.append(prompt)
    batched_prompts = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]
    return batched_prompts