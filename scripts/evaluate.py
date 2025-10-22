# scripts/evaluate.py
import argparse
import torch
from datasets import FlowClipDataset
from models import EgoGaitNet
from utils import pairwise_scores, compute_eer
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import yaml

def extract_embeddings(model, loader, device):
    model.eval()
    embs = []
    labels = []
    paths = []
    with torch.no_grad():
        for data, label, path in tqdm(loader):
            data = data.to(device)
            e = model(data, mode='embed')  # (B,D)
            embs.append(e.cpu().numpy())
            labels.extend(label.numpy())
            paths.extend(path)
    embs = np.concatenate(embs, axis=0)
    labels = np.asarray(labels)
    return embs, labels, paths

def closed_set_classification(model, loader, device):
    # Simple classification accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.to(device)
            label = label.to(device)
            logits = model(data, mode='class')
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    return correct / total

def open_set_verification(embs, labels):
    # Build pairwise scores and labels
    scores = pairwise_scores(embs, embs)  # (N,N)
    N = scores.shape[0]
    sim = []
    gt = []
    for i in range(N):
        for j in range(i+1, N):
            sim.append(scores[i,j])
            gt.append(1 if labels[i] == labels[j] else 0)
    sim = np.array(sim)
    gt = np.array(gt)
    eer, thr = compute_eer(gt, sim)
    return eer, thr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/example_config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FlowClipDataset(cfg['flows_root'])
    loader = DataLoader(dataset, batch_size=cfg.get('batch_size', 8), shuffle=False, num_workers=4)
    # load model - assume last checkpoint in cfg['ckpt_dir']
    ckpt_dir = cfg['ckpt_dir']
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    ckpts.sort()
    ckpt = ckpts[-1]
    # instantiate model with num_classes if doing closed-set
    model = EgoGaitNet(in_channels=3, base_model='r3d_18', pretrained=False,
                       cycles=cfg['cycles'], cycle_len=cfg['cycle_len'],
                       embedding_dim=cfg['embedding_dim'], num_classes=len(dataset.subject_to_idx))
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    model = model.to(device)
    # closed-set accuracy
    acc = closed_set_classification(model, loader, device)
    print(f"Closed-set classification accuracy: {acc*100:.2f}%")
    # extract embeddings
    emb_loader = DataLoader(dataset, batch_size=cfg.get('batch_size', 8), shuffle=False, num_workers=4)
    embs, labels, paths = extract_embeddings(model, emb_loader, device)
    eer, thr = open_set_verification(embs, labels)
    print(f"Open-set EER: {eer*100:.2f}% at thr={thr:.4f}")

if __name__ == '__main__':
    main()
