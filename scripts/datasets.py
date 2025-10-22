# scripts/datasets.py
import os
import numpy as np
from torch.utils.data import Dataset
import torch

class FlowClipDataset(Dataset):
    """
    Expects directory structure:
      flows_root/
        <subject_id>/
          <video>_clip000.npy
          ...
    Each .npy contains shape (T, H, W, C) where C=3 (flow_x, flow_y, magnitude)
    We return tensor shape (C, T, H, W) for 3D CNNs.
    """
    def __init__(self, flows_root, subjects=None, transform=None, mode='classification'):
        super().__init__()
        self.flows_root = flows_root
        self.transform = transform
        self.mode = mode
        self.samples = []  # (path, subject_id)
        self.subject_to_idx = {}
        for subj in sorted(os.listdir(flows_root)):
            subj_path = os.path.join(flows_root, subj)
            if not os.path.isdir(subj_path):
                continue
            if subjects is not None and subj not in subjects:
                continue
            if subj not in self.subject_to_idx:
                self.subject_to_idx[subj] = len(self.subject_to_idx)
            for f in os.listdir(subj_path):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(subj_path, f), subj))
        # shuffle can be done outside
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, subj = self.samples[idx]
        arr = np.load(path)  # shape (T, H, W, 3)
        # convert to float32, normalize per-clip
        arr = arr.astype(np.float32)
        # optionally normalize flow magnitude/scale (simple normalization)
        arr_mean = arr.mean()
        arr_std = arr.std() + 1e-6
        arr = (arr - arr_mean) / arr_std
        # transpose to (C, T, H, W)
        arr = arr.transpose(3,0,1,2)
        tensor = torch.from_numpy(arr)
        label = self.subject_to_idx[subj]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label, path

# Triplet Dataset wrapper for online triplets or semi-hard mining
class TripletFlowDataset(Dataset):
    """
    Provides samples for triplet training. Simplest version: return (anchor, pos, neg) triples
    You can implement more advanced mining outside this dataset (in the training loop).
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        # group indices by subject
        self.by_subject = {}
        for i, (_, subj) in enumerate(self.base.samples):
            self.by_subject.setdefault(subj, []).append(i)
        self.subjects = list(self.by_subject.keys())

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        anchor_tensor, anchor_label, _ = self.base[idx]
        subj = list(self.base.subject_to_idx.keys())[anchor_label]
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.by_subject[subj])
        # pick negative subj
        neg_subj = subj
        while neg_subj == subj:
            neg_subj = np.random.choice(self.subjects)
        neg_idx = np.random.choice(self.by_subject[neg_subj])
        pos_tensor, _, _ = self.base[positive_idx]
        neg_tensor, _, _ = self.base[neg_idx]
        return anchor_tensor, pos_tensor, neg_tensor, anchor_label
