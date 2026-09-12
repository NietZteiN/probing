"""GPU stage 2: linear probes trained on the NEUTRAL condition only, evaluated everywhere.

Probe recipe = Kudo et al. (2026) Table 8: a single linear layer d -> 10, cross-entropy, plain
SGD, lr 1e-3, full batch (10,000), 10,000 epochs. Kept as the default for comparability; the
`--optimizer lbfgs` alternative (scikit-learn LogisticRegression) is there for speed checks and
must give the same picture -- if it does not, that is a finding about probe fragility, not a
reason to switch silently.

One probe per (role, position, layer, seed). For a role r (the "variable of interest"), the
probe predicts r's TRUE value. Training data: neutral-train hidden states. Evaluation on every
test condition yields, per instance:
    pred        argmax class
    p_true      probability of the true value
    p_lure      probability of the lure (incongruent/irrelevant instances; else NaN)
    margin      log p_true - log p_lure
Aggregates: accuracy, lure_rate (pred == lure), lure_mass (mean p_lure), mean margin.

Selectivity (Hewitt & Liang, 2019): a control task assigns each instance a label that is a
deterministic hash of the target's NAME rather than its value. A probe that scores well on the
control is reading word identity, not value. Reported as accuracy - control accuracy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

N_CLASSES = 10


def load_cache(run_dir: Path) -> tuple[np.memmap, dict]:
    meta = json.loads((run_dir / "meta.json").read_text())
    hidden = np.load(run_dir / "hidden.npy", mmap_mode="r")
    return hidden, meta


def labels_for(meta: dict, role: str) -> np.ndarray:
    return np.array([x["values"][role] for x in meta["instances"]], dtype=np.int64)


def lures_for(meta: dict, role: str) -> np.ndarray:
    """Lure label per instance if the instance's target is `role`, else -1."""
    return np.array([(x["lure"] if (x["target"] == role and x["lure"] is not None) else -1) for x in meta["instances"]], dtype=np.int64)


def control_labels(meta: dict, role: str, seed: int = 0) -> np.ndarray:
    """Hewitt & Liang control task: label = hash(name of role) mod 10, fixed per word."""
    out = []
    for x in meta["instances"]:
        h = hashlib.sha256(f"{seed}:{x['names'][role]}".encode()).digest()
        out.append(h[0] % N_CLASSES)
    return np.array(out, dtype=np.int64)


class LinearProbe:
    def __init__(self, d: int, seed: int, device: str = "cuda"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = (torch.randn(N_CLASSES, d, generator=g) * (1.0 / np.sqrt(d))).to(device)
        self.b = torch.zeros(N_CLASSES, device=device)
        self.device = device

    def fit_sgd(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 10_000) -> list[float]:
        W = self.W.clone().requires_grad_(True); b = self.b.clone().requires_grad_(True)
        opt = torch.optim.SGD([W, b], lr=lr)
        losses = []
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(X @ W.T + b, y)
            loss.backward(); opt.step()
            if ep % 1000 == 0 or ep == epochs - 1:
                losses.append(float(loss))
        self.W, self.b = W.detach(), b.detach()
        return losses

    def fit_lbfgs(self, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X, y)
        W = np.zeros((N_CLASSES, X.shape[1]), dtype=np.float32); b = np.zeros(N_CLASSES, dtype=np.float32)
        W[clf.classes_] = clf.coef_; b[clf.classes_] = clf.intercept_
        self.W = torch.tensor(W, device=self.device); self.b = torch.tensor(b, device=self.device)

    @torch.no_grad()
    def logprobs(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(X @ self.W.T + self.b, dim=-1)


class BatchedProbes:
    """K independent linear probes trained simultaneously with the same recipe (full-batch SGD,
    cross-entropy, lr 1e-3, 10k epochs). Probe k sees X[k] : [n, d]. Numerically the same as K
    separate LinearProbe.fit_sgd runs (independent parameters, independent gradients); only the
    matmuls are batched, which makes a (position) sweep over 29 layers x 3 seeds one job of
    minutes instead of hours."""

    def __init__(self, K: int, d: int, seeds: list[int], device: str = "cuda"):
        Ws = []
        for sd in seeds:
            g = torch.Generator(device="cpu").manual_seed(sd)
            Ws.append(torch.randn(N_CLASSES, d, generator=g) * (1.0 / np.sqrt(d)))
        # layout: K = n_layers * n_seeds, seed-major within each layer
        n_layers = K // len(seeds)
        self.W = torch.stack([Ws[j] for _ in range(n_layers) for j in range(len(seeds))]).to(device)  # [K, 10, d]
        self.b = torch.zeros(K, N_CLASSES, device=device)
        self.device = device

    def fit_sgd(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 10_000) -> None:
        """X: [K, n, d] (fp16 or fp32 on device), y: [n]."""
        W = self.W.clone().requires_grad_(True); b = self.b.clone().requires_grad_(True)
        opt = torch.optim.SGD([W, b], lr=lr)
        Xf = X.float()
        yk = y.unsqueeze(0).expand(X.shape[0], -1).reshape(-1)
        K, n, _ = X.shape
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = torch.baddbmm(b.unsqueeze(1), Xf, W.transpose(1, 2))       # [K, n, 10]
            # mean over n per probe, summed over probes: each probe's gradient is its own mean loss
            loss = torch.nn.functional.cross_entropy(logits.reshape(K * n, N_CLASSES), yk, reduction="none").view(K, n).mean(1).sum()
            loss.backward(); opt.step()
        self.W, self.b = W.detach(), b.detach()

    def probe(self, k: int, d: int) -> "LinearProbe":
        lp = LinearProbe.__new__(LinearProbe)
        lp.W, lp.b, lp.device = self.W[k], self.b[k], self.device
        return lp


def evaluate(probe: LinearProbe, X: torch.Tensor, y: np.ndarray, lure: np.ndarray) -> dict:
    lp = probe.logprobs(X).cpu().numpy()
    pred = lp.argmax(-1)
    yt = y
    p_true = np.exp(lp[np.arange(len(yt)), yt])
    has_lure = lure >= 0
    p_lure = np.full(len(yt), np.nan); margin = np.full(len(yt), np.nan)
    if has_lure.any():
        idx = np.where(has_lure)[0]
        p_lure[idx] = np.exp(lp[idx, lure[idx]])
        margin[idx] = lp[idx, yt[idx]] - lp[idx, lure[idx]]
    return {
        "accuracy": float((pred == yt).mean()),
        "lure_rate": float((pred[has_lure] == lure[has_lure]).mean()) if has_lure.any() else None,
        "lure_mass": float(np.nanmean(p_lure)) if has_lure.any() else None,
        "margin_mean": float(np.nanmean(margin)) if has_lure.any() else None,
        "margin_pos_frac": float((margin[has_lure] > 0).mean()) if has_lure.any() else None,
        "per_instance": {"pred": pred.tolist(), "p_true": p_true.round(5).tolist(),
                         "p_lure": np.where(np.isnan(p_lure), None, p_lure.round(5)).tolist(),
                         "margin": np.where(np.isnan(margin), None, margin.round(4)).tolist()},
    }


def to_device(h: np.ndarray, device: str, standardize: tuple[np.ndarray, np.ndarray] | None = None) -> torch.Tensor:
    x = torch.tensor(np.asarray(h, dtype=np.float32), device=device)
    if standardize is not None:
        mu, sd = standardize
        x = (x - torch.tensor(mu, device=device)) / torch.tensor(sd, device=device)
    return x


def train_and_eval(train_dir: Path, test_dirs: dict[str, Path], role: str, out_path: Path,
                   seeds=(0, 1, 2), optimizer: str = "sgd", epochs: int = 10_000, lr: float = 1e-3,
                   layers: list[int] | None = None, positions: list[str] | None = None,
                   device: str = "cuda", control: bool = True, standardize: bool = False) -> None:
    """Sweep (position, layer, seed); write one JSON with aggregates and per-instance outputs."""
    Htr, mtr = load_cache(train_dir)
    ytr = labels_for(mtr, role)
    ctr = control_labels(mtr, role)
    tests = {c: load_cache(p) for c, p in test_dirs.items()}
    pos_labels = positions or [l for l in mtr["pos_labels"]]
    layer_idx = layers or list(range(len(mtr["layers"])))
    results = []
    per_inst: dict[str, np.ndarray] = {}
    for pl in pos_labels:
        pi = mtr["pos_labels"].index(pl)
        ytr_t = torch.tensor(ytr, device=device); ctr_t = torch.tensor(ctr, device=device)
        # ---- all layers x seeds of this position at once
        Xall = torch.tensor(np.asarray(Htr[:, pi, :, :]), device=device)          # [n, L, d] fp16
        Xall = Xall.permute(1, 0, 2).contiguous()                                   # [L, n, d]
        L, n, d = Xall.shape
        stds = None
        if standardize:
            mu = Xall.float().mean(1, keepdim=True); sd = Xall.float().std(1, keepdim=True) + 1e-6
            Xall = ((Xall.float() - mu) / sd).half(); stds = (mu, sd)
        sel = [li for li in layer_idx]
        Xsel = Xall[sel]                                                            # [Ls, n, d]
        K = len(sel) * len(seeds)
        Xk = Xsel.repeat_interleave(len(seeds), dim=0)                              # [K, n, d]
        if optimizer == "sgd":
            bp = BatchedProbes(K, d, list(seeds), device); bp.fit_sgd(Xk, ytr_t, lr=lr, epochs=epochs)
            cp = None
            if control:
                cp = BatchedProbes(K, d, list(seeds), device); cp.fit_sgd(Xk, ctr_t, lr=lr, epochs=epochs)
        for a_, li in enumerate(sel):
            for b_, seed in enumerate(seeds):
                k = a_ * len(seeds) + b_
                Xtr = Xsel[a_]
                if optimizer == "sgd":
                    probe = bp.probe(k, d)
                else:
                    Xtr_np = np.asarray(Htr[:, pi, li, :], dtype=np.float32)
                    if stds is not None:
                        Xtr_np = (Xtr_np - stds[0][a_].cpu().numpy()) / stds[1][a_].cpu().numpy()
                    probe = LinearProbe(d, seed, device); probe.fit_lbfgs(Xtr_np, ytr, seed)
                rec = {"role": role, "position": pl, "layer": int(mtr["layers"][li]), "seed": seed,
                       "train_acc": float((probe.logprobs(Xtr.float()).argmax(-1).cpu().numpy() == ytr).mean()), "eval": {}}
                for cond, (Hte, mte) in tests.items():
                    if pl not in mte["pos_labels"]:
                        continue
                    pj = mte["pos_labels"].index(pl)
                    Xte = torch.tensor(np.asarray(Hte[:, pj, li, :]), device=device).float()
                    if stds is not None:
                        Xte = (Xte - stds[0][a_]) / stds[1][a_]
                    ev = evaluate(probe, Xte, labels_for(mte, role), lures_for(mte, role))
                    pi_ = ev.pop("per_instance")
                    for field, arr in pi_.items():
                        per_inst[f"{pl}/L{mtr['layers'][li]}/s{seed}/{cond}/{field}"] = np.array(
                            [np.nan if v is None else v for v in arr], dtype=np.float32)
                    rec["eval"][cond] = ev
                if control:
                    if optimizer == "sgd":
                        cprobe = cp.probe(k, d)
                    else:
                        cprobe = LinearProbe(d, seed, device); cprobe.fit_lbfgs(np.asarray(Htr[:, pi, li, :], dtype=np.float32), ctr, seed)
                    ctl = {}
                    for cond, (Hte, mte) in tests.items():
                        if pl not in mte["pos_labels"]:
                            continue
                        pj = mte["pos_labels"].index(pl)
                        Xte = torch.tensor(np.asarray(Hte[:, pj, li, :]), device=device).float()
                        if stds is not None:
                            Xte = (Xte - stds[0][a_]) / stds[1][a_]
                        ctl[cond] = float((cprobe.logprobs(Xte).argmax(-1).cpu().numpy() == control_labels(mte, role)).mean())
                    rec["control_acc"] = ctl
                results.append(rec)
                print(f"{role} {pl} L{mtr['layers'][li]} s{seed} train={rec['train_acc']:.3f} "
                      + " ".join(f"{c}={v['accuracy']:.3f}" for c, v in rec["eval"].items()), flush=True)
        del Xall, Xsel, Xk
        torch.cuda.empty_cache() if device == "cuda" else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"train_dir": str(train_dir), "role": role, "optimizer": optimizer,
                                    "epochs": epochs, "lr": lr, "standardize": standardize,
                                    "test_ids": {c: [x["id"] for x in m["instances"]] for c, (_, m) in tests.items()},
                                    "results": results}))
    np.savez_compressed(out_path.with_suffix(".npz"), **per_inst)
