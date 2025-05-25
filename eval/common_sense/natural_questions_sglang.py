import json
from pathlib import Path
import argparse
import sglang as sgl
from datasets import load_dataset
import sys
sys.path.append(".")
from eval.prompt_utils import get_prompts
from eval.eval_utils import calculate_semantic_diversity

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate and classify text using a language model.")
parser.add_argument("--mode", type=str, default="natural", help="Prompt mode")
parser.add_argument("--style", type=str, default="news", help="Style of the text to generate")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
parser.add_argument("--num_samples", type=int, default=512, help="Number of samples to generate")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
parser.add_argument("--prompt_mode", type=str, default="natural", help="Prompt mode")
parser.add_argument("--n", type=int, default=10, help="Number of samples to generate")
args = parser.parse_args()

if __name__ == "__main__":
    mode = args.mode
    style = args.style

    dataset = load_dataset('sentence-transformers/natural-questions', split='train')
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))

    batched_prompts = get_prompts("natural_questions", dataset, args)
    llm = sgl.Engine(model_path=args.model, log_level="info")
    sampling_params = {
        "temperature": args.temperature, 
        "top_p": args.top_p, 
        "repetition_penalty": args.repetition_penalty, 
        "max_new_tokens": args.max_new_tokens,
        "n": args.n
    }
    all_outputs = []
    for prompt_batch in batched_prompts:
        responses = llm.generate(prompt_batch, sampling_params)
        batched_responses = []
        for i in range(0, len(responses), args.n):
            batched_responses.append([responses[j]['text'] for j in range(i, i+args.n)])
        all_outputs.extend(batched_responses)

    llm.shutdown()

    outputs_path = Path('data') / f"outputs_natural_questions_eval_{mode}_{args.prompt_mode}.json"
    with open(outputs_path, "w") as f:
        json.dump(all_outputs, f, indent=4)

    semantic_diversity = calculate_semantic_diversity(outputs_path)
    with open('eval/common_sense/results/natural_questions_results.txt', 'a') as f:
        f.write(f"mode: {mode}, prompt_mode: {args.prompt_mode}, temperature: {args.temperature}, semantic_diversity: {semantic_diversity}\n")
