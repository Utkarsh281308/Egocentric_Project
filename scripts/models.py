# scripts/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18  # available from torchvision

class EgoGaitNet(nn.Module):
    """
    Simplified implementation of EgoGaitNet from the paper:
      - 3D CNN backbone produces spatio-temporal features per input clip segment (gait cycle)
      - We split a clip into N gait cycles; extract per-cycle features, pass through LSTM to merge
    Assumptions:
      - Input clip has shape (B, C, T, H, W), where T = 60 frames (or configured)
      - We'll split T into M cycles each of L frames (e.g., M=4, L=15)
    """
    def __init__(self, in_channels=3, base_model='r3d_18', pretrained=True,
                 cycles=4, cycle_len=15, embedding_dim=4096, num_classes=None):
        super().__init__()
        self.cycles = cycles
        self.cycle_len = cycle_len
        self.embedding_dim = embedding_dim
        # backbone
        if base_model == 'r3d_18':
            self.backbone = r3d_18(pretrained=pretrained)
            # r3d_18 expects input channels=3; adapt conv1 if in_channels != 3
            if in_channels != 3:
                old_conv = self.backbone.stem[0]
                new_conv = nn.Conv3d(in_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride, padding=old_conv.padding,
                                     bias=old_conv.bias)
                # initialize new conv
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                self.backbone.stem[0] = new_conv
            # replace classifier with identity
            self.backbone.fc = nn.Identity()
            backbone_feat_dim = 512  # r3d_18 outputs 512-d after global pool
        else:
            raise NotImplementedError

        # After extracting spatial channels, we will expand to embedding_dim via FC
        self.fc_cycle = nn.Sequential(
            nn.Linear(backbone_feat_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # LSTM to merge M cycles -> single embedding
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim,
                            num_layers=1, batch_first=True)

        # head for classification (closed-set)
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, num_classes)
            )
        else:
            self.classifier = None

        # projection head for metric learning (optional)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward_backbone(self, x):
        # x: (B,C,T,H,W) or per-cycle chunk e.g., (B,C,L,H,W)
        # r3d_18 expects (B, C, T, H, W)
        # pass through backbone to get (B,512)
        feat = self.backbone(x)  # shape (B, 512)
        return feat

    def forward(self, x, mode='embed'):
        """
        x: (B, C, T, H, W)
        mode: 'embed' -> returns embedding (B, embedding_dim)
              'class' -> returns class logits (B, num_classes)
        """
        B, C, T, H, W = x.shape
        # split along temporal into cycles
        assert T == self.cycles * self.cycle_len, f"T should be cycles*cycle_len. Got {T}"
        cycle_feats = []
        for i in range(self.cycles):
            t0 = i * self.cycle_len
            t1 = t0 + self.cycle_len
            chunk = x[:, :, t0:t1, :, :]  # (B,C,cycle_len,H,W)
            # pass through backbone (we may need to adapt backbone which expects certain T)
            # Optionally, if backbone required larger T, adjust during training.
            # For simplicity, pass chunk directly.
            feat = self.forward_backbone(chunk)  # (B, backbone_dim)
            feat = self.fc_cycle(feat)  # (B, embedding_dim)
            cycle_feats.append(feat.unsqueeze(1))
        cycle_feats = torch.cat(cycle_feats, dim=1)  # (B, M, embedding_dim)
        # LSTM
        out, (h_n, c_n) = self.lstm(cycle_feats)  # out (B, M, embedding_dim)
        # take last output
        gait_embedding = out[:, -1, :]  # (B, embedding_dim)
        if mode == 'embed':
            # L2-normalize embedding for metric tasks
            e = self.proj(gait_embedding)
            e = F.normalize(e, p=2, dim=1)
            return e
        elif mode == 'class':
            assert self.classifier is not None, "Classifier not initialized"
            logits = self.classifier(gait_embedding)
            return logits
        else:
            raise ValueError("Unknown mode.")

# Triplet loss wrapper (can use torch.nn.TripletMarginLoss as well)
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)

# Skeleton for HSN (first-person to third-person matching). Full HSN requires third-person gait extractor.
class HybridSiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=4096):
        super().__init__()
        # Example: two LSTMs projecting modalities to same space
        self.lstm_f = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.lstm_t = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.proj_f = nn.Linear(embedding_dim, embedding_dim)
        self.proj_t = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, feat_f, feat_t):
        # feat_f: (B, M, D) first-person cycles; feat_t: (B, M, D) third-person cycles
        _, (h_f, _) = self.lstm_f(feat_f)
        _, (h_t, _) = self.lstm_t(feat_t)
        h_f = h_f[-1]
        h_t = h_t[-1]
        pf = F.normalize(self.proj_f(h_f), p=2, dim=1)
        pt = F.normalize(self.proj_t(h_t), p=2, dim=1)
        return pf, pt
