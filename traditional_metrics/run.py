import argparse
import json
import os
from tqdm import tqdm
from collections import Counter
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# nltk.download("punkt")

def distinct_n(texts, n=1):
    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:
        tokens = word_tokenize(text)
        n_gram_list = list(ngrams(tokens, n))
        unique_ngrams.update(n_gram_list)
        total_ngrams += len(n_gram_list)

    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

def self_bleu(texts, n=2):
    scores = []
    weights = tuple((1.0 / n for _ in range(n)))
    smoothing = SmoothingFunction().method1

    for i in tqdm(range(len(texts)), desc=f"Calculating self-BLEU-{n}"):
        hypothesis = word_tokenize(texts[i])
        references = [word_tokenize(texts[j]) for j in range(len(texts)) if j != i]
        if not hypothesis or not references:
            continue
        score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
        scores.append(score)
    
    return float(np.mean(scores)) if scores else 0.0

def compute_metrics(texts):
    metrics = {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "distinct_4": distinct_n(texts, 4),
        "distinct_5": distinct_n(texts, 5),
        "self_bleu_2": self_bleu(texts, 2),
        "self_bleu_3": self_bleu(texts, 3),
    }
    return metrics

def load_texts(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [d["text"] for d in data if d is not None and "text" in d]

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    texts = load_texts(args.data_file)
    metrics = compute_metrics(texts)

    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        f.write(json.dumps({"id": f"{args.mode}_{args.prompt_mode}", **metrics}) + "\n")

    print(f"\n📊 Diversity Metrics Analysis ({args.prompt_mode})")
    for k, v in metrics.items():
        print(f" - {k:20s}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, default="steer")
    parser.add_argument("--output_dir", type=str, default="traditional_metrics/results")
    parser.add_argument("--mode", type=str, default="llama")
    args = parser.parse_args()
    main(args)
