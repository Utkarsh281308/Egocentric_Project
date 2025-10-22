# scripts/train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets import FlowClipDataset, TripletFlowDataset
from models import EgoGaitNet, TripletLoss
from utils import save_checkpoint
import os
from tqdm import tqdm

def train_classification(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FlowClipDataset(cfg['flows_root'])
    num_classes = len(dataset.subject_to_idx)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    model = EgoGaitNet(in_channels=3, base_model='r3d_18', pretrained=True,
                       cycles=cfg['cycles'], cycle_len=cfg['cycle_len'],
                       embedding_dim=cfg['embedding_dim'], num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for data, labels, _ in pbar:
            data = data.to(device)  # (B,C,T,H,W)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(data, mode='class')
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/ (pbar.n+1))
        # save checkpoint
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename=os.path.join(cfg['ckpt_dir'], f'epoch_{epoch+1}.pth.tar'))
    print("Training finished.")

def train_triplet(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dataset = FlowClipDataset(cfg['flows_root'])
    trip_dataset = TripletFlowDataset(base_dataset)
    loader = DataLoader(trip_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    model = EgoGaitNet(in_channels=3, base_model='r3d_18', pretrained=True,
                       cycles=cfg['cycles'], cycle_len=cfg['cycle_len'], embedding_dim=cfg['embedding_dim'],
                       num_classes=None)
    model = model.to(device)
    criterion = TripletLoss(margin=cfg['margin'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(loader, desc=f"Triplet Epoch {epoch+1}/{cfg['epochs']}")
        for anchor, pos, neg, _ in pbar:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            optimizer.zero_grad()
            a_e = model(anchor, mode='embed')
            p_e = model(pos, mode='embed')
            n_e = model(neg, mode='embed')
            loss = criterion(a_e, p_e, n_e)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename=os.path.join(cfg['ckpt_dir'], f'trip_epoch_{epoch+1}.pth.tar'))
    print("Triplet training finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['class', 'triplet'], default='class')
    parser.add_argument('--config', default='experiments/example_config.yaml')
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.mode == 'class':
        train_classification(cfg)
    else:
        train_triplet(cfg)

if __name__ == '__main__':
    main()
