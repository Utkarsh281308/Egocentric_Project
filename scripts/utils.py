# scripts/utils.py
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import os
import math

def compute_eer(labels, scores):
    """
    labels: 1 for genuine (same), 0 for impostor
    scores: similarity scores (higher means more likely genuine)
    Compute EER by finding threshold where FAR == FRR ~
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # find point where difference between fpr and fnr is minimal
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr

def pairwise_scores(embeddings_a, embeddings_b):
    """
    Compute similarity scores (dot product) between all pairs of embeddings
    embeddings_a: (N, D)
    embeddings_b: (M, D)
    returns (N, M)
    """
    embeddings_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True)+1e-8)
    embeddings_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True)+1e-8)
    return np.dot(embeddings_a, embeddings_b.T)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    if not os.path.exists(path):
        return model, optimizer, 0
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['state_dict'])
    if optimizer and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    epoch = ck.get('epoch', 0)
    return model, optimizer, epoch
