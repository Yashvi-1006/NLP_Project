"""
Train an English stance detector on the final translator's output.

Each JSONL row (hi/bn rows come straight from build_stance_data.py):
  translation, label, lang ("en"|"hi"|"bn"),
  sarcastic (bool|null), sarcasm_target, tone, intended,
  labse_sim, sarcasm_vote_share, issues (list), low_confidence (bool)
English-native rows: leave translator fields null (best: annotate them with the same LLM).

--fields chooses which translator metadata the model sees besides `translation`:
  none | sarcastic | sarcastic,target,tone | intended | sarcastic,target,tone,intended
Each field is randomly hidden during training (--p_drop) so the model survives missing/wrong fields.

Stage 1 (English + round-trip augmented English):
  python train_stance.py --train en_train.jsonl --dev dev.jsonl --test test.jsonl --out ckpt1
Stage 2 (translated hi/bn train split, lower LR):
  python train_stance.py --train train.jsonl --dev dev.jsonl --test test.jsonl \
      --init ckpt1 --lr 1e-5 --epochs 3 --out ckpt2
"""
import argparse, json, random
import emoji, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

ALL_FIELDS = {"sarcastic", "target", "tone", "intended"}
VAGUE = {"", "none", "unclear", "unknown", "n/a", "na"}

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True)
ap.add_argument("--test", required=True);  ap.add_argument("--out", required=True)
ap.add_argument("--model", default="microsoft/deberta-v3-base")
ap.add_argument("--init", default=None, help="checkpoint dir to continue from (stage 2)")
ap.add_argument("--fields", default="sarcastic")
ap.add_argument("--p_drop", type=float, default=0.3, help="per-field dropout during training")
ap.add_argument("--skip_low_conf_train", action="store_true",
                help="drop low_confidence rows from TRAINING only (dev/test untouched)")
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=16);     ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
FIELDS = {f for f in args.fields.split(",") if f and f != "none"}
assert FIELDS <= ALL_FIELDS, f"unknown fields: {FIELDS - ALL_FIELDS}"
set_seed(args.seed); random.seed(args.seed)

load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train_rows, dev_rows, test_rows = load(args.train), load(args.dev), load(args.test)
if args.skip_low_conf_train:
    train_rows = [r for r in train_rows if not r.get("low_confidence")]

src = args.init or args.model
tok = AutoTokenizer.from_pretrained(src)
if args.init:
    model = AutoModelForSequenceClassification.from_pretrained(args.init)
    l2i = model.config.label2id
else:
    labels = sorted({r["label"] for r in train_rows + dev_rows + test_rows})
    l2i = {l: i for i, l in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label={i: l for l, i in l2i.items()}, label2id=l2i)

# ------------------------------------------------------------ input format --
def demoji(t):                       # DeBERTa/RoBERTa vocabularies handle emoji poorly -> words
    return emoji.demojize(t, delimiters=(" [", "] ")).replace("_", " ")

def prep(r, train):
    a = demoji(r["translation"]).strip()
    keep = lambda: not (train and random.random() < args.p_drop)       # per-field dropout
    s = r.get("sarcastic")
    tgt = (r.get("sarcasm_target") or "").strip()
    tone = (r.get("tone") or "").strip()
    meaning = (r.get("intended") or "").strip()
    meta = []
    if "sarcastic" in FIELDS and keep():
        meta.append("sarcastic: " + ("yes" if s else "no" if s is False else "unknown"))
    if "target" in FIELDS and s and tgt.lower() not in VAGUE and keep():
        meta.append("target: " + tgt)
    if "tone" in FIELDS and tone.lower() not in VAGUE and keep():
        meta.append("tone: " + tone)
    if "intended" in FIELDS and meaning.lower() not in {"", "unclear"} and keep():
        meta.append("meaning: " + meaning)
    return a, (" | ".join(meta) if meta else None)

def weight(r):                       # down-weight noisy translations; English-native = 1.0
    if r.get("lang", "en") == "en" or r.get("labse_sim") is None:
        return 1.0
    w = min(1.0, max(0.3, (r["labse_sim"] - 0.5) / 0.4))
    w *= 0.5 + 0.5 * (r.get("sarcasm_vote_share") or 1.0)
    if r.get("issues"):
        w *= 0.7
    return w

class DS(torch.utils.data.Dataset):
    def __init__(self, rows, train): self.rows, self.train = rows, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; a, b = prep(r, self.train)
        enc = dict(tok(a, b, truncation=True, max_length=224))
        enc["labels"] = l2i[r["label"]]
        enc["weight"] = weight(r) if self.train else 1.0
        return enc

def collate(batch):
    y = torch.tensor([b.pop("labels") for b in batch])
    w = torch.tensor([b.pop("weight") for b in batch], dtype=torch.float)
    out = tok.pad(batch, return_tensors="pt"); out["labels"], out["weight"] = y, w
    return out

counts = np.bincount([l2i[r["label"]] for r in train_rows], minlength=len(l2i)).astype(float)
class_w = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float)

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y, w = inputs.pop("labels"), inputs.pop("weight")
        out = model(**inputs)
        ce = torch.nn.functional.cross_entropy(out.logits, y, weight=class_w.to(out.logits.device),
                                               reduction="none", label_smoothing=0.05)
        loss = (ce * w.to(ce.device)).mean()
        return (loss, out) if return_outputs else loss

def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

if __name__ == "__main__":
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64, learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="linear", eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        fp16=torch.cuda.is_available(), report_to="none", remove_unused_columns=False, seed=args.seed)
    trainer = WTrainer(model=model, args=targs, train_dataset=DS(train_rows, True),
                       eval_dataset=DS(dev_rows, False), data_collator=collate, compute_metrics=metrics,
                       callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    trainer.save_model(args.out); tok.save_pretrained(args.out)

    # ---------------------------------------------- honest, sliced evaluation --
    pred = trainer.predict(DS(test_rows, False)).predictions.argmax(-1)
    gold = np.array([l2i[r["label"]] for r in test_rows])
    def rep(name, mask):
        mask = np.array(mask)
        if mask.sum() >= 5:
            print(f"{name:<26} n={mask.sum():<5} macro-F1={f1_score(gold[mask], pred[mask], average='macro'):.3f}")
    rep("ALL", [True] * len(test_rows))
    for lang in ("en", "hi", "bn"):
        rep(f"lang={lang}", [r.get("lang", "en") == lang for r in test_rows])
    rep("sarcastic=True",  [r.get("sarcastic") is True for r in test_rows])
    rep("sarcastic=False", [r.get("sarcastic") is False for r in test_rows])
    rep("low_confidence=True",  [bool(r.get("low_confidence")) for r in test_rows])
    rep("low_confidence=False", [not r.get("low_confidence") for r in test_rows])
