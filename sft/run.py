import os
import argparse
import torch
import json
import transformers
import wandb
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from huggingface_hub import whoami
import random
from dotenv import load_dotenv
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-3B', help='Model ID')
parser.add_argument('--data_path', type=str, default='/data/longfei/diversity_sft_dataset.json', help='Path to training data')
parser.add_argument('--output_dir', type=str, default='/data/longfei/ckpts/sft/llama3.2_3b_diversity', help='Output directory')
parser.add_argument('--batch_size', type=int, default=1, help='Per device batch size')
parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=6e-6, help='Learning rate')
parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
parser.add_argument('--use_flash_attn', action='store_true', help='Use Flash Attention')
parser.add_argument('--id', type=str, help='ID')
parser.add_argument('--save_total_limit', type=int, default=3, help='Total limit of saved checkpoints')
parser.add_argument('--save_steps', type=int, default=1000, help='Number of steps between checkpoint saves')
parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume from checkpoint')
parser.add_argument('--max_steps', type=int, default=10000, help='Max steps')
args = parser.parse_args()

# huggingface token
load_dotenv()
user = whoami(token=os.getenv('HF_TOKEN'))

lora_config = LoraConfig(
    r=128,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

if not args.debug:
    wandb.init(project="sft-llama3.2-3b")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
sft_model_id = args.model_id + "-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tulu_tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
all_tokenizers = [llama_tokenizer, tulu_tokenizer, mistral_tokenizer, qwen_tokenizer, phi_tokenizer]

model_kwargs = {"device_map": "auto"}
if args.use_4bit:
    model_kwargs["quantization_config"] = bnb_config
if args.use_flash_attn:
    model_kwargs["use_flash_attention_2"] = True

model = AutoModelForCausalLM.from_pretrained(
    args.model_id, 
    use_cache=not args.use_gradient_checkpointing, 
    **model_kwargs)

if args.use_gradient_checkpointing:
    model.gradient_checkpointing_enable()

# if torch.__version__ >= "2.0.0":
#     model = torch.compile(model)

if args.id == 'all_data':
    dataset = load_dataset(args.data_path, split='train')
elif args.id == 'alpaca':
    dataset = load_dataset('json', data_files=args.data_path, split='train')
else:
    if args.data_path.startswith('/data'):
        dataset = load_from_disk(args.data_path)
    else:
        dataset = load_dataset(args.data_path, split='train')
    # add a new column 'text' to the dataset by apply chat template
if args.debug:
    dataset = dataset.select(range(100))

def apply_chat_template(batch):
    texts = []
    for messages in batch["messages"]:
        text = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    batch["text"] = texts
    return batch

def apply_natural_template(batch):
    texts = []
    for messages in batch["messages"]:
        text = ""
        for message in messages:
            text += f"{message['content']}\n"
        texts.append(text.rstrip())
    
    batch["text"] = texts
    return batch

def apply_mixed_template(batch):
    texts = []
    for messages in batch["messages"]:
        # random select a tokenizer
        random_tokenizer = random.choice(all_tokenizers)
        try:
            text = random_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            print(messages)
            print(e)
            text = ""
        texts.append(text)
    
    batch["text"] = texts
    return batch

def apply_mixed_training_template(batch):
    texts = []
    sources = batch['source']
    original_texts = batch['messages']
    for source, original_text in zip(sources, original_texts):
        if source == 'fineweb':
            texts.append(original_text[0]['content'])
        else:
            text = llama_tokenizer.apply_chat_template(original_text, tokenize=False, add_generation_prompt=False)
            texts.append(text)
    batch["text"] = texts
    return batch

if args.id == 'natural' or args.id == 'natural_subset':
    dataset = dataset.map(apply_natural_template, batched=True, num_proc=16)
elif args.id == 'mixed' or args.id == 'mixed_subset':
    dataset = dataset.map(apply_mixed_template, batched=True, num_proc=16)
elif args.id == 'mixed_training' or args.id == 'mixed_training_subset':
    dataset = dataset.map(apply_mixed_training_template, batched=True, num_proc=16)
else:
    dataset = dataset.map(apply_chat_template, batched=True, num_proc=16)

tokenizer = llama_tokenizer
tokenizer.pad_token = tokenizer.eos_token
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=False), batched=True, num_proc=16)
dataset = dataset.filter(lambda x: len(x['input_ids']) <= 1024, num_proc=16)
# shuffle the dataset
# dataset = dataset.shuffle(seed=42)
print(len(dataset))

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        report_to="wandb" if not args.debug else None,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=1,
        output_dir=args.output_dir,
        logging_dir="logs",
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        max_steps=args.max_steps
    ),
    peft_config=lora_config if args.use_lora else None,
    dataset_text_field="text",
)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)