"""
Fixed training harness for ImageNet-100 (subset) architecture experiments.

    from train import init_train, train, NUM_CLASSES
    init_train()                          # one-time: load + GPU-cache the dataset
    model = build_your_model(NUM_CLASSES) # the ONLY thing you change
    metrics = train(model)                # 10 epochs, AdamW, everything else fixed

Everything except the model architecture is fixed here:
  dataset       init_train(dataset=...): 'cifar100' (100), 'imagenet-100', or 'imagewoof' (10)
  data          N_TRAIN train / N_VAL val, stratified, decoded once into a uint8
                tensor on the GPU (so steps are even and fast; no per-step decode)
  resolution    RESOLUTION (resize -> center-crop)
  augmentation  random horizontal flip (train only); val is the bare base
  optimizer     AdamW(lr=LR, wd=WEIGHT_DECAY), no decay on norm/bias
  schedule      EPOCHS epochs, linear warmup -> cosine
  precision     bf16 on Ampere+, fp16 on Turing (T4); channels_last

`train()` re-seeds the data shuffle + augmentation so different architectures see
identical batches -> a fair A/B. Only the model's weight init differs.
"""
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from datasets import load_dataset
from tqdm.auto import tqdm

# ----------------------------- fixed config ------------------------------ #
DATASET = "frgfm/imagewoof"
DATASET_CONFIG = None     # HF builder config (load_dataset's 2nd arg); imagewoof uses fast.ai tarball
NUM_CLASSES = 20          # imagewoof has 10 dog-breed classes
N_TRAIN = -1              # total train imgs, split equally across classes; -1 = whole dataset
N_VAL = -1               # total val imgs, split equally across classes;  -1 = whole dataset
RESOLUTION = 32           # set by init_train(resolution=...): 32 for CIFAR-100, 224 for full-res
BATCH_SIZE = 256          # larger batch -> better GPU utilization at 64px (tiny activations)
EPOCHS = 10 
COMPILE_BLOCKS = False     # torch.compile each residual block (GPU only); set False to disable
WARMUP_EPOCHS = 1
LR = 1e-3                 # AdamW peak
MIN_LR = 1e-5
WEIGHT_DECAY = 0.05
SEED = 0
PREPROC_WORKERS = 0       # 0 = no worker processes (fast enough here, no shutdown noise)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DEVICE = DEVICE     # hold the whole (small) dataset on the GPU
CACHE_DIR = os.path.expanduser("~/.cache/convnext_tensors")  # decoded uint8 tensors persisted here;
                                                             # point at a Drive path to survive restarts

if DEVICE.type == "cuda":                 # GPU throughput: autotune conv algos + allow TF32
    torch.backends.cudnn.benchmark = True                 # fixed input shape -> tune once, reuse
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if COMPILE_BLOCKS:                     # each block compiles once per (stage width x train/eval);
        import torch._dynamo               # raise dynamo's cap above the default 8 so every variant
        for _a in ("recompile_limit", "cache_size_limit"):   # compiles instead of falling back to
            if hasattr(torch._dynamo.config, _a):            # eager (which spams recompile warnings)
                setattr(torch._dynamo.config, _a, 64)
        torch._dynamo.config.suppress_errors = True   # a compiler bug -> eager fallback, never a crash

# Dataset presets picked by init_train(dataset=...). The chosen preset overwrites the
# DATASET / DATASET_CONFIG / NUM_CLASSES / N_TRAIN / N_VAL globals above, so a model
# built after init_train sizes its head to the right number of classes.
# imagewoof is NOT loaded via the Hub (its loader is a script, unsupported by recent
# `datasets`); _load_splits pulls fast.ai's tarball + the `imagefolder` builder instead.
DATASETS = {
    "cifar100":     dict(hf="uoft-cs/cifar100",    config=None, num_classes=100, n_train=-1, n_val=-1,
                         img_key="img",   label_key="fine_label", val_split="test"),
    "imagenet-100": dict(hf="clane9/imagenet-100", config=None, num_classes=100, n_train=-1, n_val=-1,
                         img_key="image", label_key="label",      val_split="validation"),
    "imagewoof":    dict(hf="frgfm/imagewoof",     config=None, num_classes=10,  n_train=-1, n_val=-1,
                         img_key="image", label_key="label",      val_split="validation"),
}
# HF field/split names for the current dataset (overwritten by init_train per preset)
IMG_KEY, LABEL_KEY, VAL_SPLIT = "img", "fine_label", "test"
# fast.ai's 320px imagewoof tarball (10 dog breeds, train/ + val/ of wnid folders)
IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"

# ----------------------------- module state ------------------------------ #
_S = {"ready": False, "dataset": None, "resolution": None, "Xtr": None, "Ytr": None,
      "Xva": None, "Yva": None, "amp": None, "mean": None, "std": None}


def _amp_dtype():
    if DEVICE.type != "cuda":
        return None
    # native bf16 needs Ampere+ (sm_80); Turing (T4) -> fp16. is_bf16_supported() lies on T4.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16



class _DecodeDS(Dataset):
    """Decode + resize to a fixed size; returns uint8 CxHxW and a remapped label."""
    def __init__(self, hf, idx, remap):
        self.hf, self.idx, self.remap = hf, idx, remap
        self.tf = T.Compose([
            T.Resize(int(RESOLUTION / 0.875), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(RESOLUTION),
            T.PILToTensor(),
        ])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ex = self.hf[int(self.idx[i])]
        img = ex[IMG_KEY]
        if img.mode != "RGB":
            img = img.convert("RGB")          # guard grayscale / CMYK
        return self.tf(img), self.remap[ex[LABEL_KEY]]


def _select_indices(labels, classes, n_total, seed):
    """Pick indices for `classes`, sampling every class equally.

    n_total == -1 -> keep every image of the selected classes (whole dataset).
    Otherwise take exactly n_total // len(classes) images per class (a balanced
    subset), capped at each class's availability. The total may be slightly below
    n_total when it isn't divisible by the number of classes.
    """
    rng = np.random.default_rng(seed)
    per = None if n_total == -1 else n_total // len(classes)
    chunks = []
    for c in classes:
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        chunks.append(ci if per is None else ci[:per])
    idx = np.concatenate(chunks)
    rng.shuffle(idx)
    return idx


def _build_cache(hf, idx, remap, name):
    loader = DataLoader(_DecodeDS(hf, idx, remap), batch_size=256,
                        num_workers=PREPROC_WORKERS, shuffle=False)
    xs, ys = [], []
    for xb, yb in tqdm(loader, desc=f"caching {name}"):
        xs.append(xb)
        ys.append(yb)
    return torch.cat(xs), torch.cat(ys)


def _imagewoof_split(split_dir):
    """Build an HF Dataset of {'image': PIL, 'label': int} from <wnid>/<img> folders.

    Done by hand (Dataset.from_dict + the lazy Image feature) rather than the
    `imagefolder` builder, whose split auto-detection is brittle across `datasets`
    versions. Classes are the sorted wnid folder names, so train and val get an
    identical label order (label i means the same breed in both).
    """
    import os, glob
    from datasets import Dataset, Features, Image, ClassLabel
    classes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    exts = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in sorted(glob.glob(os.path.join(split_dir, c, "*"))):
            if os.path.splitext(p)[1].lower() in exts:
                paths.append(p); labels.append(i)
    feats = Features({"image": Image(), "label": ClassLabel(names=classes)})
    return Dataset.from_dict({"image": paths, "label": labels}, features=feats)


def _load_splits(dataset):
    """Return (train_hf, val_hf) HF datasets, each yielding {'image': PIL, 'label': int}.

    imagewoof ships on the Hub as a loading *script* (imagewoof.py), which recent
    `datasets` refuses to run ("Dataset scripts are no longer supported"). We instead
    fetch fast.ai's official imagewoof2-320 tarball once and read the extracted
    <split>/<wnid>/ folders directly (see _imagewoof_split). Labels are the WordNet-id
    folder names, remapped to 0..N-1 downstream like any other dataset.
    """
    if dataset == "imagewoof":
        import os, tarfile, urllib.request
        cache = os.path.expanduser("~/.cache")
        base = os.path.join(cache, "imagewoof2-320")
        if not os.path.isdir(base):
            os.makedirs(cache, exist_ok=True)
            print("downloading imagewoof2-320 (~300 MB) from fast.ai ...", flush=True)
            tgz, _ = urllib.request.urlretrieve(IMAGEWOOF_URL)
            with tarfile.open(tgz) as t:
                t.extractall(cache)                       # -> {cache}/imagewoof2-320/{train,val}
        return (_imagewoof_split(os.path.join(base, "train")),
                _imagewoof_split(os.path.join(base, "val")))
    ds = load_dataset(DATASET, DATASET_CONFIG) if DATASET_CONFIG else load_dataset(DATASET)
    return ds["train"], ds[VAL_SPLIT]


def _disk_cache_path(dataset):
    """Path for the persisted decoded tensors, keyed by everything that changes them."""
    key = f"{dataset}_r{RESOLUTION}_c{NUM_CLASSES}_tr{N_TRAIN}_va{N_VAL}_s{SEED}.pt"
    return os.path.join(CACHE_DIR, key)


def _trim_to_batch(X, Y):
    """Drop the last partial batch so every batch is exactly BATCH_SIZE (uniform shape)."""
    m = (X.size(0) // BATCH_SIZE) * BATCH_SIZE
    return X[:m], Y[:m]


def init_train(dataset="cifar100", resolution=32, force=False):
    """Load `dataset` ('cifar100' / 'imagenet-100' / 'imagewoof') at `resolution`, cache to device.

    `resolution` sets the decode size: 32 for the CIFAR-100 tour (the models in
    expirements/ are 32x32-only, /1 stem -> 4x4 final) and 224 for a full-resolution
    run. The chosen preset (see DATASETS) overwrites DATASET/NUM_CLASSES/N_TRAIN/N_VAL
    and the field-name globals so a model built afterwards sizes its head correctly.
    Idempotent per (dataset, resolution); pass a new one (or force=True) to re-cache.
    """
    global DATASET, DATASET_CONFIG, NUM_CLASSES, N_TRAIN, N_VAL, RESOLUTION
    global IMG_KEY, LABEL_KEY, VAL_SPLIT
    assert dataset in DATASETS, f"dataset must be one of {set(DATASETS)}"
    cfg = DATASETS[dataset]
    DATASET, DATASET_CONFIG = cfg["hf"], cfg["config"]
    NUM_CLASSES, N_TRAIN, N_VAL = cfg["num_classes"], cfg["n_train"], cfg["n_val"]
    IMG_KEY, LABEL_KEY, VAL_SPLIT = cfg["img_key"], cfg["label_key"], cfg["val_split"]
    RESOLUTION = resolution

    if _S["ready"] and _S["dataset"] == dataset and _S["resolution"] == resolution and not force:
        print(f"init_train: already initialized ({dataset} @ {resolution}px, "
              f"train={_S['Xtr'].size(0)}, val={_S['Xva'].size(0)}, classes={NUM_CLASSES})")
        return NUM_CLASSES
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    cache_path = _disk_cache_path(dataset)
    if os.path.exists(cache_path) and not force:
        print(f"init_train: loading decoded cache from {cache_path}", flush=True)
        blob = torch.load(cache_path, map_location="cpu")
        Xtr, Ytr, Xva, Yva = blob["Xtr"], blob["Ytr"], blob["Xva"], blob["Yva"]
    else:
        train_hf, val_hf = _load_splits(dataset)
        train_labels = np.array(train_hf[LABEL_KEY])
        val_labels = np.array(val_hf[LABEL_KEY])
        keep = sorted(np.unique(train_labels).tolist())[:NUM_CLASSES]
        remap = {c: i for i, c in enumerate(keep)}   # labels may be ints or strings ('Beagle')

        tr_idx = _select_indices(train_labels, keep, N_TRAIN, SEED)
        va_idx = _select_indices(val_labels, keep, N_VAL, SEED + 1)
        tr_target = (N_TRAIN // NUM_CLASSES) * NUM_CLASSES   # balanced target (ignored when -1)
        va_target = (N_VAL // NUM_CLASSES) * NUM_CLASSES
        if N_TRAIN != -1 and len(tr_idx) < tr_target:
            print(f"[warn] only {len(tr_idx)} train imgs available (< {tr_target}) "
                  f"for {NUM_CLASSES} classes")
        if N_VAL != -1 and len(va_idx) < va_target:
            print(f"[warn] only {len(va_idx)} val imgs available (< {va_target}) "
                  f"for {NUM_CLASSES} classes")

        Xtr, Ytr = _build_cache(train_hf, tr_idx, remap, "train")
        Xva, Yva = _build_cache(val_hf, va_idx, remap, "val")
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({"Xtr": Xtr, "Ytr": Ytr, "Xva": Xva, "Yva": Yva}, cache_path)
        print(f"init_train: saved decoded cache -> {cache_path}", flush=True)

    # keep only whole batches so train AND val always feed the same [BATCH_SIZE, ...] shape
    # (the disk cache above stays full/untrimmed, so it's reusable if BATCH_SIZE changes)
    Xtr, Ytr = _trim_to_batch(Xtr, Ytr)
    Xva, Yva = _trim_to_batch(Xva, Yva)
    _S.update(
        Xtr=Xtr.to(CACHE_DEVICE), Ytr=Ytr.to(CACHE_DEVICE),
        Xva=Xva.to(CACHE_DEVICE), Yva=Yva.to(CACHE_DEVICE),
        amp=_amp_dtype(),
        mean=torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1),
        std=torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1),
        dataset=dataset,
        resolution=resolution,
        ready=True,
    )
    gb = _S["Xtr"].element_size() * _S["Xtr"].nelement() / 1e9
    print(f"init_train: {dataset} @ {resolution}px train {tuple(_S['Xtr'].shape)} "
          f"({gb:.2f} GB) on {_S['Xtr'].device}, val {tuple(_S['Xva'].shape)}, "
          f"classes={NUM_CLASSES}, amp={_S['amp']}")
    return NUM_CLASSES


def _lr_at(step, total, warm):
    if step < warm:
        return LR * (step + 1) / max(1, warm)
    if total == warm:
        return LR
    p = (step - warm) / (total - warm)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * p))


def _prep(xb, yb, is_train, process_batch=None):
    x = xb.to(DEVICE, non_blocking=True).float().div_(255.0)   # [B,3,H,W] in [0,1]
    y = yb.to(DEVICE, non_blocking=True)
    if is_train:                                               # fixed aug: horizontal flip
        m = torch.rand(x.size(0), device=x.device) < 0.5
        if m.any():
            x[m] = torch.flip(x[m], dims=[-1])
    x = x * 2.0 - 1.0
    if process_batch is not None:                              # user aug hook, in [-1, 1] space
        x, y = process_batch(x, y)                 # gets [-1,1] images + labels
    x = (x + 1.0) * 0.5                                    # aug output [-1,1] -> [0,1]
    x = (x - _S["mean"]) / _S["std"]
    return x.to(memory_format=torch.channels_last), y


def sample_batch(n=8, train_split=True):
    """Return `(x, y)`: `n` images as a float tensor [n, 3, H, W] normalised to [-1, 1]
    and their integer labels [n], both on DEVICE.

    Handy for previewing a `process_batch(x, y)` hook (and graph.visualize_batch, which
    expects the [-1, 1] range). Call after init_train().
    """
    assert _S["ready"], "call init_train() first"
    Xsrc, Ysrc = (_S["Xtr"], _S["Ytr"]) if train_split else (_S["Xva"], _S["Yva"])
    x = Xsrc[n:2*n].to(DEVICE, non_blocking=True).float().div_(255.0)
    y = Ysrc[n:2*n].to(DEVICE, non_blocking=True)
    return x * 2.0 - 1.0, y


def _compile_blocks(model):
    """torch.compile each residual block in place (regional compilation).

    Compiling the repeated block instead of the whole model keeps compile time low
    (one block shape is compiled once and reused across instances) and avoids
    top-level graph breaks. Blocks are found by class name ending in 'Block'
    (BasicBlock / Block); their parameters are shared with the wrapper, so the
    optimizer built afterwards still sees every parameter. dynamic=False keeps every
    compile STATIC (our shapes are fixed per stage) -- this avoids automatic-dynamic,
    whose symbolic-shape backward codegen is buggy for the LayerNorm reduction.
    """
    targets = [(parent, name) for parent in model.modules()
               for name, child in parent.named_children()
               if type(child).__name__.endswith("Block")]
    for parent, name in targets:
        parent._modules[name] = torch.compile(parent._modules[name], dynamic=False)
    return model


def _param_groups(model):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def _evaluate(model, amp):
    model.eval()
    Xva, Yva, bs = _S["Xva"], _S["Yva"], BATCH_SIZE   # same batch size as training
    top1 = torch.zeros((), device=DEVICE)
    top5 = torch.zeros((), device=DEVICE)
    n = 0
    for it in range(Xva.size(0) // bs):               # Xva trimmed to a multiple of bs -> full batches only
        x, y = _prep(Xva[it * bs:(it + 1) * bs], Yva[it * bs:(it + 1) * bs], False)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(x)
        p5 = out.topk(5, dim=1).indices
        top1 += (p5[:, 0] == y).sum()
        top5 += p5.eq(y.view(-1, 1)).any(1).sum()
        n += y.size(0)
    return 100.0 * top1.item() / n, 100.0 * top5.item() / n


def train(model, name=None, verbose=True, process_batch=None):
    """Train `model` for EPOCHS with the fixed recipe. Returns a metrics dict.

    `name` labels the experiment; it is stored in the returned metrics under
    "name" so graph() can title each run (defaults to the model's class name).

    `process_batch` (optional) is an augmentation hook run on every TRAINING batch
    (never on val). Signature: process_batch(x, y) -> (x, y). It receives images x as a
    float tensor [B, C, H, W] normalised to [-1, 1] plus integer labels y [B], and must
    return the images in the same shape/range together with labels. y may stay integer
    [B] (e.g. random-erase) or become soft targets [B, num_classes] (mixup / cutmix) --
    CrossEntropyLoss handles both. Define it in the notebook and pass it in.
    """
    assert _S["ready"], "call init_train() first"
    if name is None:
        name = type(model).__name__
    torch.manual_seed(SEED)   # fix shuffle + aug so architectures see identical batches

    Xtr, Ytr = _S["Xtr"], _S["Ytr"]
    amp = _S["amp"]
    model = model.to(DEVICE, memory_format=torch.channels_last)
    if COMPILE_BLOCKS and DEVICE.type == "cuda":
        model = _compile_blocks(model)
    opt = torch.optim.AdamW(_param_groups(model), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    crit = nn.CrossEntropyLoss()

    bs = BATCH_SIZE
    spe = Xtr.size(0) // bs                 # drop last partial batch -> uniform steps
    total, warm = EPOCHS * spe, WARMUP_EPOCHS * spe

    hist = {"epoch": [], "train_loss": [], "train_acc": [],
            "val_top1": [], "val_top5": [],
            "train_time_s": [], "val_time_s": [], "epoch_time_s": []}

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(Xtr.size(0), device=Xtr.device)
        # accumulate on the GPU; .item() once per epoch avoids a host sync every step
        loss_acc = torch.zeros((), device=DEVICE)
        correct_acc = torch.zeros((), device=DEVICE)
        seen = 0
        t0 = time.time()
        steps = range(spe)
        if verbose:
            steps = tqdm(steps, desc=f"epoch {epoch}", leave=False)
        for it in steps:
            idx = perm[it * bs:(it + 1) * bs]
            x, y = _prep(Xtr[idx], Ytr[idx], True, process_batch)
            for g in opt.param_groups:
                g["lr"] = _lr_at(epoch * spe + it, total, warm)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                out = model(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_acc += loss.detach() * x.size(0)
            y_hard = y if y.ndim == 1 else y.argmax(1)         # soft labels (mixup/cutmix) -> top class
            correct_acc += (out.detach().argmax(1) == y_hard).sum()
            seen += x.size(0)
        loss_sum = loss_acc.item(); correct = correct_acc.item()   # sync -> train time accurate
        train_dt = time.time() - t0

        tv = time.time()
        v1, v5 = _evaluate(model, amp)   # ends in .item() -> already synced, val time accurate
        val_dt = time.time() - tv

        hist["epoch"].append(epoch)
        hist["train_loss"].append(loss_sum / seen)
        hist["train_acc"].append(100.0 * correct / seen)
        hist["val_top1"].append(v1)
        hist["val_top5"].append(v5)
        hist["train_time_s"].append(train_dt)
        hist["val_time_s"].append(val_dt)
        hist["epoch_time_s"].append(train_dt + val_dt)
        if verbose:
            print(f"epoch {epoch:2d} | train_loss {loss_sum/seen:.3f} | "
                  f"train_acc {100*correct/seen:5.2f} | val_top1 {v1:5.2f} | "
                  f"val_top5 {v5:5.2f} | train {train_dt:4.1f}s | val {val_dt:4.1f}s", flush=True)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    metrics = dict(hist)
    metrics.update(
        name=name,
        best_val_top1=max(hist["val_top1"]),
        final_val_top1=hist["val_top1"][-1],
        params_M=params_m,
        total_train_time_s=sum(hist["train_time_s"]),
        total_val_time_s=sum(hist["val_time_s"]),
        total_time_s=sum(hist["epoch_time_s"]),
    )
    if verbose:
        print(f"== [{name}] best val_top1 {metrics['best_val_top1']:.2f} | "
              f"final {metrics['final_val_top1']:.2f} | {params_m:.1f}M params | "
              f"train {metrics['total_train_time_s']:.0f}s + val "
              f"{metrics['total_val_time_s']:.0f}s = {metrics['total_time_s']:.0f}s ==")
    return metrics
