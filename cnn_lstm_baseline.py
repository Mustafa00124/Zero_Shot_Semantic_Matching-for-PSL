# pip install torch torchvision opencv-python
import os, random, cv2, torch, torch.nn as nn, torch.optim as optim
import json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class VideoISLRDataset(Dataset):
    def __init__(self, root, clip_len=24, size=299):
        self.samples = []  # (video_path, class_idx)
        self.classes = []
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        # Nested layout support
        for c in [d for d in entries if os.path.isdir(os.path.join(root, d))]:
            self.class_to_idx.setdefault(c, len(self.classes))
            if c not in self.classes:
                self.classes.append(c)
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        # Flat layout support
        for f in entries:
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath) and f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                stem = os.path.splitext(f)[0]
                self.class_to_idx.setdefault(stem, len(self.classes))
                if stem not in self.classes:
                    self.classes.append(stem)
                self.samples.append((fpath, self.class_to_idx[stem]))

        self.clip_len = clip_len
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def _read_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_indices(self, n):
        if n <= self.clip_len:
            idx = list(range(n))
            # loop last frame if short
            idx += [n-1]*(self.clip_len-n)
            return idx
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_frames(vp)
        idxs = self._sample_indices(len(frames))
        clip = torch.stack([self.t(frames[j]) for j in idxs], dim=0)  # [T,3,H,W]
        return clip, y

    def __len__(self): return len(self.samples)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        # keep everything up to the final pooling (2048-d)
        self.backbone = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out_dim = 2048

    def forward(self, x):                   # x: [B,3,299,299]
        f = self.backbone(x)                # [B,2048,1,1]
        return torch.flatten(f, 1)          # [B,2048]

class CNNLSTM(nn.Module):
    def __init__(self, feat_dim=2048, hidden=512, num_classes=10):
        super().__init__()
        self.feat = InceptionFeatureExtractor()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, clip):                # clip: [B,T,3,299,299]
        B,T,_,_,_ = clip.shape
        x = clip.view(B*T, 3, 299, 299)
        with torch.no_grad():               # freeze CNN for a quick baseline
            f = self.feat(x)                # [B*T,2048]
        f = f.view(B, T, -1)
        out, _ = self.lstm(f)               # [B,T,H]
        logits = self.head(out[:, -1])      # last timestep
        return logits


def _remap_subset(samples):
    mapping = {}
    next_id = 0
    remapped = []
    for path, y in samples:
        if y not in mapping:
            mapping[y] = next_id
            next_id += 1
        remapped.append((path, mapping[y]))
    return remapped, next_id


def run_cnn_lstm(num_words=1, root="Words", seed: int = 42, out_dir: str = "results"):
    ds = VideoISLRDataset(root, clip_len=24, size=299)
    if len(ds) == 0:
        print(f"No videos found in {root}")
        return
    rng = random.Random(seed)
    indices = list(range(len(ds.samples)))
    rng.shuffle(indices)
    indices = indices[:max(1, num_words)]
    subset = [ds.samples[i] for i in indices]
    subset, num_classes = _remap_subset(subset)

    class _Subset(Dataset):
        def __init__(self, parent, samples):
            self.parent = parent
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            path, y = self.samples[i]
            frames = self.parent._read_frames(path)
            idxs = self.parent._sample_indices(len(frames))
            clip = torch.stack([self.parent.t(frames[j]) for j in idxs], dim=0)
            return clip, y

    subset_ds = _Subset(ds, subset)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM(num_classes=num_classes).to(device)
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for clips, y in loader:
            clips = clips.to(device); y = torch.tensor(y).to(device)
            logits = model(clips)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    acc = (correct/total) if total>0 else 0.0
    print(f"CNN-LSTM subset accuracy on {total} samples: {acc:.3f}")

    # Save results
    method_dir = os.path.join(out_dir, "cnn_lstm")
    os.makedirs(method_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(method_dir, f"accuracy_seed{seed}_n{num_words}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": ts,
            "method": "cnn_lstm",
            "seed": seed,
            "num_words": num_words,
            "total": int(total),
            "correct": int(correct),
            "accuracy": acc
        }, f, indent=2)
    print(f"Saved results -> {out_path}")
    return {"accuracy": acc, "total": total, "correct": correct}

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for clips, y in loader:
        clips, y = clips.to(device), torch.tensor(y).to(device)
        opt.zero_grad()
        logits = model(clips)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*clips.size(0)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred==y).sum().item()
    return loss_sum/total, correct/total

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--root", type=str, default="Words")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    run_cnn_lstm(num_words=args.num_words, root=args.root, seed=args.seed, out_dir=args.out_dir)
