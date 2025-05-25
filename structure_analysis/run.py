import json
import argparse
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from tqdm import tqdm
import math

# nltk.download("punkt")
# nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def shannon_entropy(values, num_bins=10):
    hist, _ = np.histogram(values, bins=num_bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-10))

def compute_metrics(texts):
    lengths = []
    num_sentences = []
    content_ratios = []

    for text in tqdm(texts):
        if not text.strip():
            continue

        tokens = word_tokenize(text)
        lengths.append(len(tokens))

        sentences = sent_tokenize(text)
        num_sentences.append(len(sentences))

        content_words = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]
        stopword_count = len([w for w in tokens if w.lower() in stop_words])
        if len(tokens) > 0:
            content_ratio = len(content_words) / (len(tokens) - stopword_count + 1e-6)
        else:
            content_ratio = 0.0
        content_ratios.append(content_ratio)

    def summarize(name, values):
        arr = np.array(values)
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_entropy": float(shannon_entropy(arr)),
        }

    metrics = {}
    metrics.update(summarize("token_length", lengths))
    metrics.update(summarize("sentence_count", num_sentences))
    metrics.update(summarize("content_ratio", content_ratios))
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
        f.write(json.dumps({"id":f"{args.mode}_{args.prompt_mode}", **metrics}) + "\n")

    print(f"\n📊 Structure Pattern Analysis ({args.prompt_mode})")
    for k, v in metrics.items():
        print(f" - {k:30s}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, default="steer")
    parser.add_argument("--output_dir", type=str, default="structure_analysis/results")
    parser.add_argument("--mode", type=str, default="llama")
    if not os.path.exists(parser.parse_args().output_dir):
        os.makedirs(parser.parse_args().output_dir)
    args = parser.parse_args()
    main(args)
