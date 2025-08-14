# pip install torch torchvision opencv-python
import os, cv2, torch, torch.nn as nn, torch.optim as optim
import random, json, time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoISLR3D(Dataset):
    def __init__(self, root, clip_len=16, size=112):
        self.samples = []  # List[Tuple[path, class_idx]]
        self.classes = []  # List[str]
        self.class_to_idx = {}

        entries = sorted(os.listdir(root))
        # Collect directory-structured classes
        dir_classes = [d for d in entries if os.path.isdir(os.path.join(root, d))]
        for c in dir_classes:
            self.class_to_idx.setdefault(c, len(self.classes))
            if c not in self.classes:
                self.classes.append(c)
            cdir = os.path.join(root, c)
            for f in os.listdir(cdir):
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        # Also support flat layout: each .mp4 is its own class (video name)
        for f in entries:
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath) and f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                stem = os.path.splitext(f)[0]
                self.class_to_idx.setdefault(stem, len(self.classes))
                if stem not in self.classes:
                    self.classes.append(stem)
                self.samples.append((fpath, self.class_to_idx[stem]))

        self.clip_len = clip_len
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989])  # Kinetics-400 stats

    def _read_rgb(self, path):
        cap = cv2.VideoCapture(path); frames=[]
        while True:
            ok, f = cap.read()
            if not ok: break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release(); return frames

    def _sample_idx(self, n):
        if n <= self.clip_len: return list(range(n)) + [n-1]*(self.clip_len-n)
        step = n / self.clip_len
        return [int(i*step) for i in range(self.clip_len)]

    def __getitem__(self, i):
        vp, y = self.samples[i]
        frames = self._read_rgb(vp)
        idxs = self._sample_idx(len(frames))
        clip = []
        for j in idxs:
            img = self.to_tensor(frames[j])         # [3,H,W]
            img = self.resize(img)                  # [3,112,112]
            img = self.norm(img)
            clip.append(img)
        x = torch.stack(clip, dim=1)                # [3,T,H,W]
        return x, y

    def __len__(self): return len(self.samples)

def make_model(num_classes, pretrained=True):
    if pretrained:
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        m = r3d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

class TinyC3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(64,128,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(128,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(256,256,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(256,512,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512,512,3,padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):              # x: [B,3,T,112,112]
        f = self.features(x)           # [B,512,1,1,1]
        return self.classifier(f.view(x.size(0), -1))


def _remap_subset(samples):
    """Remap labels in given sample list to a compact 0..K-1 space."""
    old_to_new = {}
    new_samples = []
    next_idx = 0
    for path, y in samples:
        if y not in old_to_new:
            old_to_new[y] = next_idx
            next_idx += 1
        new_samples.append((path, old_to_new[y]))
    return new_samples, next_idx


def run_c3d(num_words=1, root="Words_train", use_pretrained=True, seed: int = 42, out_dir: str = "results"):
    """Run a minimal C3D pipeline on a small subset (num_words samples) and save accuracy."""
    ds = VideoISLR3D(root, clip_len=16, size=112)
    if len(ds) == 0:
        print(f"No videos found in {root}")
        return

    # Select subset by seed
    rng = random.Random(seed)
    indices = list(range(len(ds.samples)))
    rng.shuffle(indices)
    indices = indices[:max(1, num_words)]
    subset_samples = [ds.samples[i] for i in indices]
    subset_samples, num_classes = _remap_subset(subset_samples)

    # Build a tiny ad-hoc dataset object
    class _Subset(Dataset):
        def __init__(self, parent, samples):
            self.parent = parent
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            # Monkey-patch parent access pattern
            path, y = self.samples[i]
            frames = parent = self.parent
            frames = parent._read_rgb(path)
            idxs = parent._sample_idx(len(frames))
            clip = []
            for j in idxs:
                img = parent.to_tensor(frames[j])
                img = parent.resize(img)
                img = parent.norm(img)
                clip.append(img)
            x = torch.stack(clip, dim=1)
            return x, y

    subset_ds = _Subset(ds, subset_samples)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(num_classes, pretrained=use_pretrained).to(device)
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    acc = (correct/total) if total>0 else 0.0
    print(f"C3D subset accuracy on {total} samples: {acc:.3f}")

    # Save results
    method_dir = os.path.join(out_dir, "c3d")
    os.makedirs(method_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(method_dir, f"accuracy_seed{seed}_n{num_words}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": ts,
            "method": "c3d",
            "seed": seed,
            "num_words": num_words,
            "total": int(total),
            "correct": int(correct),
            "accuracy": acc
        }, f, indent=2)
    print(f"Saved results -> {out_path}")
    return {"accuracy": acc, "total": total, "correct": correct}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_words", type=int, default=1)
    ap.add_argument("--root", type=str, default="Words_train")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    run_c3d(num_words=args.num_words, root=args.root, use_pretrained=not args.no_pretrained, seed=args.seed, out_dir=args.out_dir)
