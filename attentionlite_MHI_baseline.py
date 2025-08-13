"""
PSL Attention‑Lite (MHI‑guided) for Isolated Sign Recognition
-------------------------------------------------------------
Minimal PyTorch implementation that injects a spatial attention mask derived
from a Motion History Image (MHI) into a 3D CNN (R3D‑18) mid‑network.

Folder layout expected:
  data/
    train/
      CLASS_A/*.mp4
      CLASS_B/*.mp4
      ...
    val/
      CLASS_A/*.mp4
      CLASS_B/*.mp4

Usage:
  python psl_attention_lite_mhi_r3d18.py --train_dir data/train --val_dir data/val

Notes:
- We compute an RGB‑MHI for visualization/optionally auxiliary stream, and a
  grayscale MHI heatmap used as attention. The attention is injected after
  layer2 of R3D‑18: F := F * (1 + alpha * A), where A is the resized MHI mask
  broadcast over time and channels.
- Keep clips short (T=16) at 112×112 for speed.
- Requires: torch, torchvision, opencv-python
"""
import os
import argparse
import cv2
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights

# ---------------------------
# Utility: video loading
# ---------------------------

def read_video_cv2(path: str) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))  # HWC uint8
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read video: {path}")
    return frames

# ---------------------------
# MHI computation
# ---------------------------

def compute_rgb_mhi(frames: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (rgb_mhi[3,H,W] float in [0,1], gray_mhi[1,H,W] float in [0,1]).
    RGB‑MHI packs motion from early/mid/late thirds into B,G,R channels.
    Gray MHI is the sum (normalized), used for attention.
    """
    # convert to grayscale float32
    grays = [cv2.cvtColor(f.numpy(), cv2.COLOR_RGB2GRAY).astype('float32') for f in frames]
    H, W = grays[0].shape
    # frame diffs -> positive motion
    diffs = []
    for i in range(1, len(grays)):
        d = cv2.absdiff(grays[i], grays[i-1])
        diffs.append(d)
    if len(diffs) == 0:
        diffs = [grays[0].copy()]

    # split indices into three roughly equal parts
    n = len(diffs)
    thirds = [ (0, max(1, n//3)), (max(1, n//3), max(2, 2*n//3)), (max(2, 2*n//3), n) ]
    channels = []
    for (a,b) in thirds:
        seg = diffs[a:b]
        if len(seg) == 0:
            ch = torch.zeros((H,W), dtype=torch.float32)
        else:
            acc = torch.from_numpy(seg[0])
            for k in range(1, len(seg)):
                acc = torch.maximum(acc, torch.from_numpy(seg[k]))
            ch = acc
        # normalize per channel to [0,1]
        m = float(ch.max()) if float(ch.max()) > 1e-6 else 1.0
        ch = (ch / m)
        channels.append(ch)
    # order: B,G,R (early, mid, late)
    bgr = torch.stack(channels, dim=0)
    # gray attention mask as mean of channels
    gray = bgr.mean(dim=0, keepdim=True)
    return bgr, gray

# ---------------------------
# Dataset
# ---------------------------
class ISLRDataset(Dataset):
    def __init__(self, root: str, clip_len: int = 16, size: int = 112):
        super().__init__()
        self.samples = []  # (path, class_idx)
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for c in self.classes:
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith(('.mp4','.mov','.avi','.mkv')):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))
        self.clip_len = clip_len
        self.resize = transforms.Resize((size, size), antialias=True)
        # Kinetics normalization
        self.norm = transforms.Normalize([0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989])

    def __len__(self):
        return len(self.samples)

    def _uniform_indices(self, n: int) -> List[int]:
        if n <= self.clip_len:
            return list(range(n)) + [n-1] * (self.clip_len - n)
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        frames = read_video_cv2(path)  # list of HWC uint8 tensors
        inds = self._uniform_indices(len(frames))
        sel = [frames[i] for i in inds]

        # Build 3D clip tensor [3,T,H,W]
        rgb_frames = []
        for f in sel:
            img = f.permute(2,0,1).float() / 255.0  # [3,H,W]
            img = self.resize(img)
            img = self.norm(img)
            rgb_frames.append(img)
        clip = torch.stack(rgb_frames, dim=1)  # [3,T,H,W]

        # Compute MHI (using the selected frames only for alignment)
        rgb_mhi, gray_mhi = compute_rgb_mhi(sel)
        # resize gray_mhi to input spatial size for consistency
        gray_mhi = transforms.Resize((clip.shape[-2], clip.shape[-1]), antialias=True)(gray_mhi)
        # normalize safety
        gray_mhi = torch.clamp(gray_mhi, 0.0, 1.0)

        return clip, gray_mhi, y

# ---------------------------
# Model with Attention‑Lite
# ---------------------------
class MHIAttentionR3D(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 0.5, pretrained: bool = True):
        super().__init__()
        if pretrained:
            base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            base = r3d_18(weights=None)
        # keep stem and layers, replace fc
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)
        self.alpha = alpha

    def forward(self, x: torch.Tensor, mhi_gray: torch.Tensor):
        """
        x: [B,3,T,H,W], mhi_gray: [B,1,H,W] in [0,1]
        We inject attention after layer2. We resize mhi to feature spatial size and
        broadcast over time & channels: A -> [B,1,T,Hf,Wf]. Then F := F * (1 + alpha*A).
        """
        B, C, T, H, W = x.shape
        f = self.stem(x)       # -> [B,64,T/2,H/2,W/2]
        f = self.layer1(f)     # -> [B,64,...]
        f = self.layer2(f)     # -> [B,128,...]
        # build attention volume
        Hf, Wf = f.shape[-2], f.shape[-1]
        A = torch.nn.functional.interpolate(mhi_gray, size=(Hf, Wf), mode='bilinear', align_corners=False)  # [B,1,Hf,Wf]
        A = A.unsqueeze(2).repeat(1, 1, f.shape[2], 1, 1)  # [B,1,Tf,Hf,Wf]
        f = f * (1.0 + self.alpha * A)
        # continue network
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)    # [B,512,1,1,1]
        logits = self.fc(f.flatten(1))
        return logits

# ---------------------------
# Training / Eval
# ---------------------------

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for clip, mhi, y in loader:
            clip = clip.to(device)
            mhi = mhi.to(device)
            y = torch.as_tensor(y, device=device)
            logits = model(clip, mhi)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / max(1, total)


def train(args):
    train_ds = ISLRDataset(args.train_dir, clip_len=args.clip_len, size=args.size)
    val_ds   = ISLRDataset(args.val_dir,   clip_len=args.clip_len, size=args.size)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MHIAttentionR3D(num_classes=len(train_ds.classes), alpha=args.alpha, pretrained=not args.no_pretrained).to(device)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(optim_params, lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0; count = 0; correct = 0; total = 0
        for clip, mhi, y in train_ld:
            clip = clip.to(device)
            mhi = mhi.to(device)
            y = torch.as_tensor(y, device=device)

            opt.zero_grad()
            logits = model(clip, mhi)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item()) * y.size(0)
            count += y.size(0)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        train_loss = running / max(1, count)
        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_ld, device)
        sched.step()
        print(f"Epoch {ep:02d}/{args.epochs}  loss {train_loss:.4f}  acc {train_acc:.3f}  val_acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'classes': train_ds.classes,
                'args': vars(args),
            }, os.path.join(args.out_dir, 'best_mhi_attn_r3d18.pt'))
    print(f"Best val acc: {best_acc:.3f}")


def run_attentionlite_mhi(num_words=1, root='Words', seed: int = 42, out_dir: str = 'results'):
    # Build a tiny dataset using the same class, but taking first num_words samples and remapping labels
    class _Args: pass
    args = _Args()
    args.train_dir = root
    args.val_dir = root
    args.clip_len = 16
    args.size = 112
    args.batch_size = 1
    args.workers = 0
    args.epochs = 1
    args.lr = 1e-4
    args.wd = 1e-4
    args.alpha = 0.5
    args.no_pretrained = False
    args.out_dir = 'checkpoints'

    # Create full dataset to access classes and samples
    full = ISLRDataset(root, clip_len=args.clip_len, size=args.size)
    if len(full) == 0:
        print(f"No videos found in {root}")
        return
    # Build manual subset
    import random, json, time
    rng = random.Random(seed)
    idxs = list(range(len(full.samples)))
    rng.shuffle(idxs)
    idxs = idxs[:max(1, num_words)]
    subset = [full.samples[i] for i in idxs]
    # Remap labels 0..K-1
    mapping = {}
    next_id = 0
    remapped = []
    for path, y in subset:
        if y not in mapping:
            mapping[y] = next_id; next_id += 1
        remapped.append((path, mapping[y]))

    # Minimal loader for subset
    class _Subset(Dataset):
        def __init__(self, parent, samples):
            self.parent = parent; self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            path, y = self.samples[i]
            # reuse full dataset logic
            frames = read_video_cv2(path)
            inds = self.parent._uniform_indices(len(frames))
            sel = [frames[k] for k in inds]
            # clip
            rgb_frames = []
            for f in sel:
                img = f.permute(2,0,1).float()/255.0
                img = self.parent.resize(img)
                img = self.parent.norm(img)
                rgb_frames.append(img)
            clip = torch.stack(rgb_frames, dim=1)
            _, gray = compute_rgb_mhi(sel)
            gray = transforms.Resize((clip.shape[-2], clip.shape[-1]), antialias=True)(gray)
            gray = torch.clamp(gray, 0.0, 1.0)
            return clip, gray, y

    subset_ds = _Subset(full, remapped)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MHIAttentionR3D(num_classes=next_id, alpha=args.alpha, pretrained=True).to(device)
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for clip, mhi, y in loader:
            clip, mhi, y = clip.to(device), mhi.to(device), torch.as_tensor(y, device=device)
            logits = model(clip, mhi)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    acc = (correct/total) if total>0 else 0.0
    print(f"AttentionLite MHI subset accuracy on {total} samples: {acc:.3f}")

    # Save results
    method_dir = os.path.join(out_dir, 'attentionlite_mhi')
    os.makedirs(method_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(method_dir, f"accuracy_seed{seed}_n{num_words}.json")
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': ts,
            'method': 'attentionlite_mhi',
            'seed': seed,
            'num_words': num_words,
            'total': int(total),
            'correct': int(correct),
            'accuracy': acc
        }, f, indent=2)
    print(f"Saved results -> {out_path}")
    return {"accuracy": acc, "total": total, "correct": correct}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--num_words', type=int, default=1)
    p.add_argument('--root', type=str, default='Words')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='results')
    args = p.parse_args()
    run_attentionlite_mhi(num_words=args.num_words, root=args.root, seed=args.seed, out_dir=args.out_dir)