#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bentcls/train.py

Fixes:
- Use build_model() from bentcls.models (instead of nonexistent build_resnet18_binary)
- Adds 2-class confusion matrix (BT=1, nonBT=0) as an evaluation metric
- DOES NOT write any files (no checkpoints, no CSV logs, no mkdir)
"""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .utils import seed_all, load_yaml
from .datasets import BentClsDataset, load_label_csv
from .models import build_model


@torch.no_grad()
def eval_metrics(model, loader, device, thr=0.5):
    """
    Returns:
      loss, acc, prec, rec, f1, (tn, fp, fn, tp)
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    ys, ps = [], []
    loss_sum, n = 0.0, 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(x).squeeze(1)
        loss = bce(logit, y)

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        p = torch.sigmoid(logit).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(p)

    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps)
    pred = (p >= float(thr)).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    return (loss_sum / max(n, 1)), acc, prec, rec, f1, (tn, fp, fn, tp)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    seed_all(int(cfg.get("seed", 42)))

    # ---- data ----
    df = load_label_csv(
        cfg["label_csv"],
        cfg["img_root"],
        drop_uncertain=cfg.get("drop_uncertain", True),
    )
    print("[INFO] usable samples:", len(df))
    print(df["bent"].value_counts())

    # split
    rng = np.random.RandomState(int(cfg.get("seed", 42)))
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * float(cfg.get("val_frac", 0.2))))
    df_val = df.iloc[idx[:n_val]].copy()
    df_tr  = df.iloc[idx[n_val:]].copy()
    print("[INFO] train:", len(df_tr), "val:", len(df_val))

    ds_tr = BentClsDataset(df_tr, cfg["img_root"], augment=True)
    ds_va = BentClsDataset(df_val, cfg["img_root"], augment=False)

    # imbalance handling
    y_tr = df_tr["bent"].astype(int).values
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    w_pos = n_neg / max(n_pos, 1)

    # sampler: oversample positives
    weights = np.where(y_tr == 1, w_pos, 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), sampler=sampler,
                       num_workers=int(cfg.get("num_workers", 2)), pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["batch_size"]), shuffle=False,
                       num_workers=int(cfg.get("num_workers", 2)), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # ---- model ----
    model_name = cfg.get("model_name", "resnet18")
    pretrained = bool(cfg.get("pretrained", True))
    model = build_model(model_name=model_name, pretrained=pretrained).to(device)
    print(f"[INFO] model: {model_name} pretrained={pretrained}")

    # loss with pos_weight (recall-friendly)
    pos_weight = torch.tensor([w_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))

    thr = float(cfg.get("thr", 0.5))
    best_f1 = -1.0
    best_ep = -1
    bad = 0
    patience = int(cfg.get("patience", 10))

    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for x, y, _ in tqdm(dl_tr, desc=f"Epoch {ep}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logit = model(x).squeeze(1)
            loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        tr_loss = loss_sum / max(n, 1)

        val_loss, acc, prec, rec, f1, (tn, fp, fn, tp) = eval_metrics(model, dl_va, device, thr=thr)
        print(
            f"[E{ep:03d}] tr={tr_loss:.4f} va={val_loss:.4f} "
            f"acc={acc:.3f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
            f"thr={thr:.2f} lr={lr_now:.2e} "
            f"CM[[{tn},{fp}],[{fn},{tp}]]"
        )

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_ep = ep
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] best F1={best_f1:.3f} @ epoch {best_ep}")
                break

    print(f"[DONE] best F1={best_f1:.3f} @ epoch {best_ep} (no files were written)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.config)
