from __future__ import annotations
import csv, itertools, random, time, typing
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report,
)

import spacy

SEED                = 42
MAX_CONTENT_WORDS   = 80
WORD_WINDOW         = 2      # 0 ⇒ disable word‑word edges
BATCH_SIZE          = 4
HIDDEN_DIM          = 64
EPOCHS              = 50
PATIENCE            = 5      # early‑stop patience
LR                  = 1e-3
WEIGHT_DECAY        = 5e-4
EMBED_DIM           = 300    # spaCy "en_core_web_md" vectors
EPSILON             = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1, help="number of random seeds")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--word-cap", type=int, default=MAX_CONTENT_WORDS)
    p.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--pool", choices=["entity", "all"], default="entity",
                   help="entity-focused pooling or all-node mean")
    p.add_argument("--use-ew", action="store_true", help="use entity–word edges")
    p.add_argument("--use-ee", action="store_true", help="use entity–entity edges")
    p.add_argument("--window", type=int, default=WORD_WINDOW,
                   help="word–word window size; 0 disables")
    return p.parse_args()


def coverage_report(graphs, name: str) -> None:
    n = len(graphs)
    n_with_ent = sum(int(g.has_entities.item()) for g in graphs)
    n_sw_fb    = sum(int(g.used_stopword_fallback.item()) for g in graphs)
    avg_nodes  = sum(g.num_nodes for g in graphs) / max(n, 1)
    print(
        f"[coverage] {name:5s} | docs={n:5d} | "
        f"with_entities={n_with_ent:5d} ({n_with_ent/n:5.1%}) | "
        f"stopword_fallback={n_sw_fb:5d} ({n_sw_fb/n:5.1%}) | "
        f"avg_nodes={avg_nodes:4.1f}"
    )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dedup(seq: typing.Iterable[str]) -> list[str]:
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(seq))


try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    raise SystemExit(
        "spaCy model 'en_core_web_md' is required.\n"
        "Install via:  python -m spacy download en_core_web_md"
    )


def load_20ng():
    tr = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    te = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes")
    )
    return tr.data, tr.target.tolist(), te.data, te.target.tolist(), tr.target_names

# ---------------------------------------------------------------------------
def build_graph(text: str, label: int) -> Data:
    doc = NLP(text)

    ents = dedup([e.text for e in doc.ents])
    ent_spans = {(e.start, e.end) for e in doc.ents}

    words: list[str] = []
    for tok in doc:
        if not tok.is_alpha or tok.is_stop:
            continue
        if any(s <= tok.i < e for s, e in ent_spans):
            continue
        words.append(tok.text.lower())
    words = dedup(words[:MAX_CONTENT_WORDS])

    used_stopword_fallback = False
    if not words:
        words = dedup([tok.text.lower() for tok in doc if tok.is_alpha])[:MAX_CONTENT_WORDS]
        used_stopword_fallback = True

    nodes = dedup(ents + words)

    if not nodes:
        nodes = ["<empty>"]  # extremely rare

    x_fp16 = torch.stack(
        [torch.tensor(NLP(tok).vector, dtype=torch.float16) for tok in nodes]
    )
    is_entity = torch.tensor([n in ents for n in nodes], dtype=torch.bool)

    idx = {tok: i for i, tok in enumerate(nodes)}
    edges: set[tuple[int, int]] = set()

    if ents and words:
        for e in ents:
            ei = idx[e]
            for w in words:
                wi = idx[w]
                edges.add((ei, wi)); edges.add((wi, ei))

    for sent in doc.sents:
        s_ents = [e.text for e in sent.ents if e.text in idx]
        for a, b in itertools.permutations(s_ents, 2):
            edges.add((idx[a], idx[b]))

    if WORD_WINDOW > 0 and words:
        for i in range(len(words) - WORD_WINDOW + 1):
            win = words[i:i + WORD_WINDOW]
            for u, v in itertools.permutations(win, 2):
                edges.add((idx[u], idx[v]))

    if not edges:
        i0 = 0
        edges.add((i0, i0))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    has_entities = torch.tensor([bool(ents)], dtype=torch.bool)
    used_sw_fb = torch.tensor([used_stopword_fallback], dtype=torch.bool)

    return Data(
        x=x_fp16,
        edge_index=edge_index,
        y=torch.tensor([label]),
        is_entity=is_entity,
        has_entities=has_entities,
        used_stopword_fallback=used_sw_fb,
    )


def make_loaders(batch_size: int = BATCH_SIZE):
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, label_names = load_20ng()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_raw, y_tr_raw, test_size=0.10,
        stratify=y_tr_raw, random_state=SEED
    )

    def to_graphs(xs, ys):
        graphs = []
        for x, y in zip(xs, ys):
            g = build_graph(x, y)   
            graphs.append(g)
        return graphs

    tr_ds  = to_graphs(X_tr,  y_tr)
    val_ds = to_graphs(X_val, y_val)
    te_ds  = to_graphs(X_te_raw, y_te_raw)

    coverage_report(tr_ds,  "train")
    coverage_report(val_ds, "val")
    coverage_report(te_ds,  "test")

    return (
        DataLoader(tr_ds,  batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(te_ds,  batch_size=batch_size),
        label_names,
    )



class EWGCN(torch.nn.Module):
    def __init__(self, hidden: int, n_cls: int):
        super().__init__()
        self.g1  = GCNConv(EMBED_DIM, hidden)
        self.g2  = GCNConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, n_cls)

    def forward(self, x_fp16: Tensor, ei: Tensor,
                batch: Tensor, mask: Tensor) -> Tensor:
        x = x_fp16.float()
        x = self.g1(x, ei).relu()
        x = self.g2(x, ei).relu()

        # entity‑mean pooling
        ent_sum = global_add_pool(x * mask.unsqueeze(-1).float(), batch)
        ent_cnt = global_add_pool(mask.unsqueeze(-1).float(), batch)
        doc_vec = ent_sum / (ent_cnt + EPSILON)

        # fallback if graph has zero entities (rare)
        zero = (ent_cnt.squeeze(-1) < EPSILON)
        if zero.any():
            doc_vec = torch.where(
                zero.unsqueeze(-1),
                global_mean_pool(x, batch),
                doc_vec
            )

        return torch.log_softmax(self.out(doc_vec), dim=1)


def evaluate(model: EWGCN, loader: DataLoader) -> dict[str, typing.Any]:
    model.eval()
    preds, gold = [], []
    with torch.inference_mode():
        for b in loader:
            b = b.to(DEVICE)
            out = model(b.x, b.edge_index, b.batch, b.is_entity)
            preds.append(out.argmax(1).cpu())
            gold.append(b.y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(gold).numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    return {
        "accuracy": acc,
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "micro_f1": micro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> None:
    report = classification_report(
        y_true, y_pred, target_names=label_names,
        output_dict=True, zero_division=0
    )
    with open("results_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in label_names:
            r = report[cls]
            w.writerow([
                cls,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                int(r['support']),
            ])


def train(model: EWGCN, tr_loader: DataLoader, val_loader: DataLoader):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_acc, impatience = None, 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in tr_loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = F.nll_loss(
                model(b.x, b.edge_index, b.batch, b.is_entity), b.y
            )
            loss.backward()
            opt.step()

        val_acc = evaluate(model, val_loader)["accuracy"]
        print(
            f"Epoch {epoch:02d}  val‑acc {val_acc*100:5.2f}%"
        )

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = model.state_dict()
            impatience = 0
        else:
            impatience += 1
        if impatience >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def main() -> None:
    t0 = time.time()
    set_seed()

    print("Loading data & constructing graphs …")
    tr_loader, val_loader, te_loader, labels = make_loaders()
    print(f"Graphs — train: {len(tr_loader.dataset)}, "
          f"val: {len(val_loader.dataset)}, "
          f"test: {len(te_loader.dataset)}")

    model = EWGCN(HIDDEN_DIM, n_cls=20).to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining …")
    train(model, tr_loader, val_loader)

    print("\nEvaluating …")
    val_metrics  = evaluate(model, val_loader)
    test_metrics = evaluate(model, te_loader)

    per_class_report(
        test_metrics["y_true"], test_metrics["y_pred"], labels
    )

    def pretty(d: dict[str, float]) -> str:
        keys = [
            "accuracy", "macro_precision",
            "macro_recall", "macro_f1", "micro_f1",
        ]
        return " | ".join(
            f"{k}: {d[k]*100:6.2f}%" for k in keys
        )

    print("\n=== VALIDATION (10 % hold‑out) ===")
    print(pretty(val_metrics))

    print("\n=== TEST (official split) ===")
    print(pretty(test_metrics))
    print("Per‑class F1 saved to results_per_class.csv")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} min on {DEVICE}")


if __name__ == "__main__":
    main()
