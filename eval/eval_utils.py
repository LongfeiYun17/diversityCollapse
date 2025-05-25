import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

def calculate_semantic_diversity(data_file):
    """计算文本集合的语义多样性（基于 Sentence-BERT 向量间的余弦距离）"""
    # 获取所有文本的句向量
    with open(data_file, 'r') as f:
        data = json.load(f)
    texts = []
    for items in data:
        if items is None:
            continue
        texts.append(items)
    embeddings = np.array([embedding_model.encode(batch, normalize_embeddings=True) for batch in tqdm(texts, desc="Encoding Texts")])
    # 计算余弦距离（Cosine Distance）
    distances = []
    for group in embeddings:
        distance = np.mean(pdist(group, metric='cosine'))
        distances.append(distance)
    mean_cosine_distance = np.mean(distances)  # 取平均值，作为语义多样性的指标

    return mean_cosine_distance

def calculate_label_entropy(labels):
    label_counts = Counter(labels)
    label_probs = np.array(list(label_counts.values())) / len(labels)
    return entropy(label_probs)