"""
Build stance-training rows from native-annotated Hindi/Bengali posts using the FINAL translator.

Input CSV columns:  id,text,label
  label = stance chosen by a NATIVE speaker reading the ORIGINAL post (never the translation).

Usage:
  python build_stance_data.py annotated.csv out_dir [--dev 0.1] [--test 0.2] [--seed 42]

Order of operations (this order prevents leakage):
  1. de-duplicate on a normalised key; drop keys that carry conflicting labels
  2. keep only Hindi/Bengali native-script posts
  3. split stratified by (lang, label)  <- BEFORE any translation
  4. translate each split; write out_dir/{train,dev,test}.jsonl (resumable)
"""
import argparse, csv, json, os, random, re, unicodedata
from collections import Counter, defaultdict

import free_translator_fixed as ft      # the final translator file

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")


def dedupe_key(text):
    """Lossy key to catch retweets/near-copies. Keeps letters AND combining marks (matras)."""
    t = unicodedata.normalize("NFC", text).lower()
    t = MENTION.sub("", URL.sub("", t))
    t = re.sub(r"^\s*rt\s+", "", t)              # retweet prefix
    t = "".join(c for c in t if unicodedata.category(c)[0] not in "PSC")
    return re.sub(r"\s+", " ", t).strip()


def clean_rows(rows):
    seen, conflicts = {}, set()
    for r in rows:
        k = dedupe_key(r["text"])
        if not k:
            continue
        if k in seen:
            if seen[k]["label"] != r["label"]:
                conflicts.add(k)
            continue
        seen[k] = r
    kept = []
    for k, r in seen.items():
        if k in conflicts:
            continue
        r = dict(r, lang=ft.detect_lang(r["text"]))
        if r["lang"] in ("hi", "bn"):
            kept.append(r)
    print(f"input {len(rows)} -> kept {len(kept)} "
          f"({len(conflicts)} conflicting-label keys dropped, rest duplicates/non-hi-bn)")
    return kept


def make_splits(rows, dev=0.1, test=0.2, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lang"], r["label"])].append(r)
    rng = random.Random(seed)
    out = {"train": [], "dev": [], "test": []}
    for g in groups.values():
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_dev = max(1, round(n * dev)) if n >= 3 else 0
        out["test"] += g[:n_test]
        out["dev"] += g[n_test:n_test + n_dev]
        out["train"] += g[n_test + n_dev:]
    return out


def read_rows(path):
    """Read the annotated CSV tolerantly: BOM, any header capitalisation, optional id column."""
    with open(path, encoding="utf-8-sig", newline="") as f:      # utf-8-sig strips an Excel BOM
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} is empty")
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        missing = {"text", "label"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing column(s) {sorted(missing)}; found {reader.fieldnames}")
        rows = []
        for n, r in enumerate(reader):
            text, label = (r.get("text") or "").strip(), (r.get("label") or "").strip()
            if not text or not label:
                continue                                          # skip blank / unlabeled rows
            rows.append({"id": (r.get("id") or "").strip() or f"row{n}", "text": text, "label": label})
    ids = Counter(r["id"] for r in rows)
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        raise ValueError(f"duplicate ids in CSV, e.g. {dup[:3]}; ids must be unique")
    print(f"read {len(rows)} labelled rows; labels: {dict(Counter(r['label'] for r in rows))}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path"); ap.add_argument("out_dir")
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = read_rows(a.csv_path)
    splits = make_splits(clean_rows(rows), a.dev, a.test, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    for name, items in splits.items():
        print(name, len(items), dict(Counter((r["lang"], r["label"]) for r in items)))
        path = os.path.join(a.out_dir, f"{name}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
        with open(path, "a", encoding="utf-8") as out:
            for r in items:
                if r["id"] in done:
                    continue
                res = ft.translate(r["text"])
                res.update(id=r["id"], label=r["label"], lang=r["lang"])
                out.write(json.dumps(res, ensure_ascii=False) + "\n"); out.flush()


if __name__ == "__main__":
    main()
