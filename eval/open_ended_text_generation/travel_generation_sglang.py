import os
import json
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from huggingface_hub import whoami
import argparse
import sglang as sgl
import multiprocessing as mp
from dotenv import load_dotenv
import sys
sys.path.append(".")
from eval.eval_utils import calculate_label_entropy
from eval.prompt_utils import get_prompts

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate and classify text using a language model.")
parser.add_argument("--model_engine", type=str, default="gpt-4o", help="Model engine to use")
parser.add_argument("--mode", type=str, default="natural", help="Prompt mode")
parser.add_argument("--style", type=str, default="news", help="Style of the text to generate")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
parser.add_argument("--prompt_mode", type=str, default="mixed", help="Prompt mode")
parser.add_argument("--n", type=int, default=1, help="Number of samples to generate")
args = parser.parse_args()

#user = whoami(token=os.getenv('HUGGING_FACE_READ_KEY'))

# Set up OpenAI API key and model engine
load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(api_key=api_key, timeout=30, max_retries=3)
model_engine = args.model_engine

def process_output(output):
    template = (
        f"Return the city name in the following text:\n\n{output}\n\n"
        "The answer should be formalized as \"X\" without explanation. If there is no city name, return \"None\"."
    )
    # Get classification response from OpenAI
    response = client.responses.create(
        model=model_engine,
        input=template,
        temperature=0.0
    ).output[0].content[0].text.strip()
    class_name = response.lower().strip("")
    return {
        "label": class_name,
        "text": output
    }

# Define prompt mode and style
if __name__ == "__main__":
    mode = args.mode
    style = args.style

    dataset = range(args.num_samples)
    batched_prompts = get_prompts("travel_generation", dataset, args)
    # flatten the batched prompts
    prompts = [prompt for batch in batched_prompts for prompt in batch]
    llm = sgl.Engine(model_path=args.model, log_level="info")
    sampling_params = {
        "temperature": args.temperature, 
        "top_p": args.top_p, 
        "repetition_penalty": args.repetition_penalty, 
        "max_new_tokens": args.max_new_tokens,
        "n": args.n
    }
    all_outputs = []
    responses = llm.generate(prompts, sampling_params)
    for response in responses:
        all_outputs.append(response['text'])

    llm.shutdown()

    outputs_path = Path('data') / f"outputs_travel_generation_eval_{mode}_{args.prompt_mode}.json"
    with open(outputs_path, "w") as f:
        json.dump(all_outputs, f, indent=4)
    labels = []

    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_output, all_outputs), total=len(all_outputs)))

    outputs_path = Path('data') / f"outputs_with_labels_travel_eval_{mode}_{args.prompt_mode}.json"
    with open(outputs_path, "w") as f:
        json.dump(results, f, indent=4)
    for result in results:
        if result:
            labels.append(result['label'])
    with open('eval/open_ended_text_generation/results/travel_generation_results.txt', 'a') as f:
        f.write(f'mode: {mode}, prompt_mode: {args.prompt_mode}, temperature: {args.temperature}, entropy: {calculate_label_entropy(labels)}\n')
