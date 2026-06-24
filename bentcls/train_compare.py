# bentcls/train_compare.py
# Train multiple classification backbones with the SAME split, save per-model best checkpoints,
# and write a compare_models.csv.
#
# Improvements (2026-02):
# - Add threshold sweep to report best_acc and best_f1 with their best thresholds.
# - Add threshold-free metrics: AUROC and Average Precision (AP).
# - Report confusion matrices for @0.5, best_acc_thr, and best_f1_thr.
#
# Usage (from project root /home/caojie/work/Galaxy-Morphology/bent_catalog):
#   python -m bentcls.train_compare

import os

# =========================
# Force PyTorch model cache to shared disk (only if user hasn't set TORCH_HOME already)
# NOTE: must be set BEFORE torchvision downloads weights.
# =========================
os.environ.setdefault("TORCH_HOME", "/shared/main/caojie/torch_cache")

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .utils import seed_all, ensure_dir, load_yaml, save_json
from .datasets import BentClsDataset, load_label_csv
from .models import build_model


# ---------- metrics helpers (robust without sklearn) ----------
def _roc_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC via rank statistic (equivalent to Mann-Whitney U).
    Returns nan if only one class present.
    """
    y_true = y_true.astype(int)
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # handle ties: average rank for ties
    # (simple tie handling)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average Precision (area under precision-recall curve), step-wise integral.
    Returns nan if no positive samples.
    """
    y_true = y_true.astype(int)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / n_pos

    # AP = sum over points where y==1 of precision at that point / n_pos
    ap = float((precision[y_sorted == 1]).sum() / n_pos)
    # recall is implicit in that formulation
    return ap


try:
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
    from sklearn.metrics import average_precision_score as _sk_average_precision_score

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        # sklearn throws if only one class
        try:
            return float(_sk_roc_auc_score(y_true, y_score))
        except Exception:
            return float("nan")

    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        try:
            return float(_sk_average_precision_score(y_true, y_score))
        except Exception:
            return float("nan")

except Exception:
    def roc_auc_score(y_true, y_score):
        return _roc_auc_score_numpy(np.asarray(y_true), np.asarray(y_score))

    def average_precision_score(y_true, y_score):
        return _average_precision_numpy(np.asarray(y_true), np.asarray(y_score))


def _confusion_from_probs(y: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= float(thr)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tn, fp, fn, tp


def _metrics_from_probs(y: np.ndarray, p: np.ndarray, thr: float):
    tn, fp, fn, tp = _confusion_from_probs(y, p, thr)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1, (tn, fp, fn, tp)


def sweep_thresholds(y: np.ndarray, p: np.ndarray, thrs: np.ndarray):
    """
    Returns dict with best_acc, best_acc_thr (+cm),
    best_f1, best_f1_thr (+cm)
    """
    best_acc = -1.0
    best_acc_thr = 0.5
    best_acc_cm = (0, 0, 0, 0)

    best_f1 = -1.0
    best_f1_thr = 0.5
    best_f1_cm = (0, 0, 0, 0)

    # tie-breakers:
    # - for best_acc: prefer higher acc, then higher f1, then higher thr (slightly conservative)
    # - for best_f1: prefer higher f1, then higher acc, then thr closest to 0.5 (stable)
    for thr in thrs:
        acc, prec, rec, f1, cm = _metrics_from_probs(y, p, float(thr))

        if (acc > best_acc + 1e-12) or (abs(acc - best_acc) <= 1e-12 and f1 > _metrics_from_probs(y, p, best_acc_thr)[3] + 1e-12) \
           or (abs(acc - best_acc) <= 1e-12 and abs(f1 - _metrics_from_probs(y, p, best_acc_thr)[3]) <= 1e-12 and thr > best_acc_thr):
            best_acc = acc
            best_acc_thr = float(thr)
            best_acc_cm = cm

        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and acc > _metrics_from_probs(y, p, best_f1_thr)[0] + 1e-12) \
           or (abs(f1 - best_f1) <= 1e-12 and abs(acc - _metrics_from_probs(y, p, best_f1_thr)[0]) <= 1e-12 and abs(thr - 0.5) < abs(best_f1_thr - 0.5)):
            best_f1 = f1
            best_f1_thr = float(thr)
            best_f1_cm = cm

    return {
        "best_acc": float(best_acc),
        "best_acc_thr": float(best_acc_thr),
        "best_acc_tn": int(best_acc_cm[0]),
        "best_acc_fp": int(best_acc_cm[1]),
        "best_acc_fn": int(best_acc_cm[2]),
        "best_acc_tp": int(best_acc_cm[3]),
        "best_f1": float(best_f1),
        "best_f1_thr": float(best_f1_thr),
        "best_f1_tn": int(best_f1_cm[0]),
        "best_f1_fp": int(best_f1_cm[1]),
        "best_f1_fn": int(best_f1_cm[2]),
        "best_f1_tp": int(best_f1_cm[3]),
    }


@torch.no_grad()
def eval_collect(model, loader, device):
    """
    Collect logits->probs and labels; also return average BCEWithLogitsLoss (unweighted).
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    ys, ps = [], []
    loss_sum, n = 0.0, 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logit = model(x).squeeze(1)
        loss = bce(logit, y)
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)

        p = torch.sigmoid(logit).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(p)

    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps).astype(float)
    return loss_sum / max(n, 1), y, p


def train_one_model(cfg: dict, model_name: str, pretrained: bool, run_dir: str):
    ensure_dir(run_dir)
    seed_all(cfg["seed"])

    # ---- data
    df = load_label_csv(cfg["label_csv"], cfg["img_root"], drop_uncertain=cfg.get("drop_uncertain", True))
    rng = np.random.RandomState(cfg["seed"])
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * cfg["val_frac"]))
    df_val = df.iloc[idx[:n_val]].copy()
    df_tr = df.iloc[idx[n_val:]].copy()

    ds_tr = BentClsDataset(df_tr, cfg["img_root"], augment=True)
    ds_va = BentClsDataset(df_val, cfg["img_root"], augment=False)

    # imbalance
    y_tr = df_tr["bent"].astype(int).values
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    w_pos = n_neg / max(n_pos, 1)

    # sampler: oversample positives
    weights = np.where(y_tr == 1, w_pos, 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)

    batch = int(cfg["batch_size"])
    dl_tr = DataLoader(ds_tr, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model
    model = build_model(model_name=model_name, pretrained=pretrained).to(device)

    # loss with pos_weight (recall-friendly)
    pos_weight = torch.tensor([w_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))

    best_f1_at05 = -1.0
    best_epoch = 0
    bad = 0

    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "train_log.csv")

    # threshold sweep settings
    thr_min = float(cfg.get("thr_min", 0.05))
    thr_max = float(cfg.get("thr_max", 0.95))
    thr_step = float(cfg.get("thr_step", 0.01))
    thrs = np.arange(thr_min, thr_max + 1e-12, thr_step, dtype=float)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,acc@0.5,prec@0.5,rec@0.5,f1@0.5,auroc,ap,best_acc,best_acc_thr,best_f1,best_f1_thr,lr\n")

    t0 = time.time()
    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for x, y, _ in dl_tr:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)

            logit = model(x).squeeze(1)
            loss = criterion(logit, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        tr_loss = loss_sum / max(n, 1)

        val_loss, yv, pv = eval_collect(model, dl_va, device)

        # threshold-free metrics
        auroc = roc_auc_score(yv, pv)
        ap = average_precision_score(yv, pv)

        # metrics @0.5
        acc05, prec05, rec05, f105, cm05 = _metrics_from_probs(yv, pv, 0.5)

        # sweep thresholds for best_acc/best_f1
        sweep = sweep_thresholds(yv, pv, thrs)

        with open(log_path, "a") as f:
            f.write(
                f"{ep},{tr_loss:.6f},{val_loss:.6f},"
                f"{acc05:.6f},{prec05:.6f},{rec05:.6f},{f105:.6f},"
                f"{auroc:.6f},{ap:.6f},"
                f"{sweep['best_acc']:.6f},{sweep['best_acc_thr']:.4f},"
                f"{sweep['best_f1']:.6f},{sweep['best_f1_thr']:.4f},"
                f"{lr_now:.8e}\n"
            )

        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "model_name": model_name, "pretrained": pretrained},
            last_path,
        )

        # early-stop criterion: keep your original (F1@0.5) for continuity
        if f105 > best_f1_at05 + 1e-4:
            best_f1_at05 = f105
            best_epoch = ep
            bad = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "model_name": model_name, "pretrained": pretrained},
                best_path,
            )
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break

    secs = time.time() - t0

    # evaluate best checkpoint again for stable summary
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    val_loss, yv, pv = eval_collect(model, dl_va, device)
    auroc = roc_auc_score(yv, pv)
    ap = average_precision_score(yv, pv)

    acc05, prec05, rec05, f105, cm05 = _metrics_from_probs(yv, pv, 0.5)
    tn05, fp05, fn05, tp05 = cm05

    sweep = sweep_thresholds(yv, pv, thrs)

    summary = {
        "model": model_name,
        "pretrained": int(pretrained),
        "best_epoch": int(best_epoch),

        # @0.5
        "val_loss": float(val_loss),
        "val_acc@0.5": float(acc05),
        "val_prec@0.5": float(prec05),
        "val_rec@0.5": float(rec05),
        "val_f1@0.5": float(f105),
        "tn@0.5": int(tn05),
        "fp@0.5": int(fp05),
        "fn@0.5": int(fn05),
        "tp@0.5": int(tp05),

        # threshold-free
        "auroc": float(auroc),
        "ap": float(ap),

        # best over thresholds
        **sweep,

        "seconds": float(secs),
        "best_path": best_path,
        "log_path": log_path,
        "n_train": int(len(df_tr)),
        "n_val": int(len(df_val)),
        "n_pos_train": int((df_tr["bent"] == 1).sum()),
        "n_neg_train": int((df_tr["bent"] == 0).sum()),
        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
        "thr_min": float(thr_min),
        "thr_max": float(thr_max),
        "thr_step": float(thr_step),
    }

    save_json(os.path.join(run_dir, "summary.json"), summary)
    return summary


def main(cfg_path: str, models_list: str, pretrained: bool = True):
    cfg = load_yaml(cfg_path)
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    model_names = [m.strip() for m in models_list.split(",") if m.strip()]
    results = []

    print("[INFO] config:", cfg_path)
    print("[INFO] out_dir:", out_dir)
    print("[INFO] models:", model_names)
    print("[INFO] pretrained:", pretrained)
    print("[INFO] TORCH_HOME:", os.environ.get("TORCH_HOME", ""))

    for name in model_names:
        run_dir = os.path.join(out_dir, f"cmp_{name}_pre{int(pretrained)}")
        print(f"\n========== {name} (pretrained={pretrained}) ==========")
        res = train_one_model(cfg, name, pretrained, run_dir)

        print(
            f"[DONE] {name} "
            f"acc@0.5={res['val_acc@0.5']:.3f}  "
            f"F1@0.5={res['val_f1@0.5']:.3f}  "
            f"best_acc={res['best_acc']:.3f}@{res['best_acc_thr']:.2f}  "
            f"best_f1={res['best_f1']:.3f}@{res['best_f1_thr']:.2f}  "
            f"AUROC={res['auroc']:.3f}  AP={res['ap']:.3f}  "
            f"best_epoch={res['best_epoch']}"
        )
        results.append(res)

    df = pd.DataFrame(results).sort_values(
        ["best_f1", "best_acc", "auroc", "ap", "val_f1@0.5", "val_acc@0.5"], ascending=False
    )
    out_csv = os.path.join(out_dir, "compare_models.csv")
    df.to_csv(out_csv, index=False)

    print("\n[SUMMARY] saved:", out_csv)
    print(
        df[
            [
                "model", "pretrained",
                "val_acc@0.5", "val_prec@0.5", "val_rec@0.5", "val_f1@0.5",
                "best_acc", "best_acc_thr", "best_f1", "best_f1_thr",
                "auroc", "ap",
                "best_epoch", "seconds",
            ]
        ]
    )


# =========================
# HARD-CODED SETTINGS
# =========================
if __name__ == "__main__":
    CFG_PATH = "configs/default.yaml"

    # NOTE:
    # - Keep ViT out by default (ViT expects 224x224; your images are 128x128).
    # - If you want ViT, you must also modify BentClsDataset to support resizing to 224.
    MODELS = "resnet18,efficientnet_b0,convnext_tiny"

    PRETRAINED = True
    main(CFG_PATH, MODELS, PRETRAINED)
