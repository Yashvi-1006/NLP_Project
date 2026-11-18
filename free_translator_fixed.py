"""
Hindi/Bengali -> English sarcasm-aware translator.

Pipeline:
  1. Detect native Hindi (Devanagari) or Bengali script.
  2. Normalize text while preserving punctuation/emojis for the LLM.
  3. Generate a literal IndicTrans2 baseline.
  4. Generate one greedy and several sampled LLM candidates.
  5. Rank candidates with cross-lingual LaBSE similarity; expose disagreement.
  6. Reject candidates whose fields contradict each other (e.g. sarcastic but no target).
  7. Flag uncertain outputs for review.

The primary downstream field is `translation`. `intended`, `sarcastic`,
`sarcasm_target`, and `tone` are auxiliary metadata.
"""
import json
import os
import re
import unicodedata
from collections import Counter

import emoji
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ------------------------------------------------------------------ config --
LLM_NAME = "google/gemma-2-9b-it"  # alternative: "Qwen/Qwen2.5-7B-Instruct"
MT_NAME = "ai4bharat/indictrans2-indic-en-1B"
N_SAMPLES = 4
LOW_SIM = 0.70  # calibrate on a native-speaker-verified validation set
ADAPTER_DIR = os.getenv("TRANSLATOR_ADAPTER_DIR") or None

DEVICE = "cuda"  # every model runs on GPU 0; there is no CPU fallback by design


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. In Colab use Runtime > Change runtime type > T4 GPU, "
            "then restart the runtime."
        )


LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}
FLORES = {"hi": "hin_Deva", "bn": "ben_Beng"}

# ----------------------------------------------------------- preprocessing --
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")


def detect_lang(text: str) -> str:
    """Detect the dominant supported native script; Latin-only input is unsupported."""
    text = unicodedata.normalize("NFC", text)
    dev = sum("\u0900" <= c <= "\u097F" for c in text)
    ben = sum("\u0980" <= c <= "\u09FF" for c in text)
    if dev == 0 and ben == 0:
        return "unk"
    return "hi" if dev >= ben else "bn"


def _split_hashtag(match):
    tag = match.group(1).replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tag)


def clean_for_llm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = URL.sub("", text)
    text = MENTION.sub("@user", text)
    text = HASHTAG.sub(_split_hashtag, text)
    # Keep emojis, repeated punctuation and ellipses: they can signal tone.
    return re.sub(r"\s+", " ", text).strip()


def clean_for_mt(text: str) -> str:
    return emoji.replace_emoji(clean_for_llm(text), "").strip()


def chunk(text: str, max_chars: int = 350):
    """Split only the MT baseline; the LLM receives the complete 1-2 sentence post."""
    parts = re.split(r"(?<=[।\.\?!])\s+", text)
    out, cur = [], ""
    for part in parts:
        if len(cur) + len(part) > max_chars and cur:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    return out + [cur] if cur else out


# ------------------------------------------------------- stage 1: baseline --
_mt = {}


def baseline_translate(text: str, lang: str) -> str:
    if lang not in FLORES:
        raise ValueError(f"Unsupported language code for IndicTrans2: {lang}")
    from IndicTransToolkit import IndicProcessor

    if not _mt:
        _require_cuda()
        _mt["tok"] = AutoTokenizer.from_pretrained(MT_NAME, trust_remote_code=True)
        _mt["m"] = AutoModelForSeq2SeqLM.from_pretrained(
            MT_NAME, trust_remote_code=True, torch_dtype=torch.float16
        ).to(DEVICE).eval()
        _mt["ip"] = IndicProcessor(inference=True)

    pieces = chunk(clean_for_mt(text))
    if not pieces:
        return ""
    batch = _mt["ip"].preprocess_batch(
        pieces, src_lang=FLORES[lang], tgt_lang="eng_Latn"
    )
    enc = _mt["tok"](
        batch, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(DEVICE)
    with torch.no_grad():
        out = _mt["m"].generate(
            **enc, num_beams=5, max_length=256, use_cache=False
        )
    decoded = _mt["tok"].batch_decode(out, skip_special_tokens=True)
    return " ".join(_mt["ip"].postprocess_batch(decoded, lang="eng_Latn"))


# ----------------------------------------------------------- stage 2: LLM --
SYSTEM = """You are a careful Hindi/Bengali-to-English translator for short
(one- or two-sentence) climate-change social-media posts. The English translation
will be passed to an English stance-detection model. Your most important task is
to preserve exactly what the author communicates, including stance and tone.

Work only from the supplied source sentence(s) and literal MT hint. No earlier
conversation or external context is available. Do not invent missing context.

TRANSLATION RULES:
1. Translate the source faithfully into natural English. Do not produce a summary.
2. Preserve sarcasm, irony, mockery, rhetorical questions, exaggeration,
   understatement and emotional intensity when they are present. Make the English
   wording sound comparably sarcastic to a fluent English reader, but do not add
   sarcasm markers or intensifiers that the source does not support.
3. If the source is sincere, translate it sincerely. Never assume a climate-related
   statement is sarcastic just because of its topic.
4. Preserve negation, who/what is being praised or criticized, and the author's
   support or opposition. Sarcasm may target climate-change denial, climate action,
   government policy, corporations, or another target; do not assume which.
5. Translate idioms by intended meaning while preserving their attitude.
6. Keep names, numbers, factual claims and relevant quoted phrases. Do not add facts,
   explanations, political opinions or claims that are absent from the source.
7. The literal machine translation is only a reference and may miss idioms or sarcasm.
   Correct it when needed, but do not change the source meaning.
8. If the source is genuinely ambiguous without context, preserve that ambiguity.
   Do not guess the author's intended stance. Mark the target or intent as unclear.

FIELD RULES (the five fields must agree with each other):
- SARCASTIC: yes only if the author says the opposite of what they mean or ridicules
  something; otherwise no.
- SARCASM_TARGET: if SARCASTIC is yes, name the specific person, group, claim or policy
  being mocked, e.g. "people who deny climate change" or "banning plastic straws as a
  climate solution". Never write none for a sarcastic post; write unclear only if the
  target is genuinely ambiguous (rule 8). If SARCASTIC is no, write none.
- TONE: one or two words for the author's attitude. A sarcastic post is never "neutral"
  or "sincere": use words like mocking, ironic, bitter, frustrated or dismissive. For
  non-sarcastic posts use words like concerned, hopeful, angry, informative or neutral.
- INTENDED: one plain sentence stating what the author actually believes about climate
  change or climate action (whether it is real or serious, and whether they support or
  oppose the measure discussed). State the belief itself; do not write "sarcastically"
  and do not describe the joke. If the post does not reveal the author's view on climate
  change, say what it does reveal and add "stance on climate change unclear". Write just
  "unclear" only if nothing can be determined.

Return exactly these five fields, one per line, and nothing else:
TRANSLATION: <natural English translation preserving the source's tone>
SARCASTIC: <yes or no>
SARCASM_TARGET: <specific target if sarcastic; none if not sarcastic; unclear only if genuinely ambiguous>
TONE: <one or two words; never neutral or sincere for a sarcastic post>
INTENDED: <one plain sentence stating the author's real belief about climate change or climate action>
"""


def _q(lang, src, base):
    return f"Language: {LANG_NAME[lang]}\nSource: {src}\nLiteral MT hint: {base}"


def _a(tr, sar, target, tone, intended):
    return (
        f"TRANSLATION: {tr}\n"
        f"SARCASTIC: {sar}\n"
        f"SARCASM_TARGET: {target}\n"
        f"TONE: {tone}\n"
        f"INTENDED: {intended}"
    )


# Each item: (lang, source, literal_mt, translation, sarcastic, target, tone, intended)
# Replace/extend with native-speaker-verified examples from your own data.
FEWSHOT = [
    ("hi", "वाह! दिल्ली की ज़हरीली हवा में सांस लेना कितना सुकून भरा है, क्लाइमेट चेंज तो बस अफ़वाह है।",
     "Wow! How comforting it is to breathe Delhi's toxic air, climate change is just a rumor.",
     "Oh wow, breathing Delhi's poisonous air is so relaxing. And of course, climate change is just a 'rumor'.",
     "yes", "people who claim climate change is just a rumor", "mocking",
     "The author believes climate change is real and serious, as Delhi's toxic air shows, and rejects the claim that it is a rumor."),
    ("hi", "हां हां, इलेक्ट्रिक कार चलाने से पूरी दुनिया का तापमान ठंडा हो जाएगा, क्या बात है!",
     "Yes yes, driving an electric car will cool the temperature of the whole world, what a thing!",
     "Oh yeah, sure, driving an electric car will totally cool down the whole planet. Brilliant!",
     "yes", "the idea that electric cars alone can fix climate change", "mocking",
     "The author thinks electric cars alone will not meaningfully cool the planet; stance on climate change itself unclear."),
    ("hi", "जलवायु परिवर्तन के कारण ग्लेशियर तेज़ी से पिघल रहे हैं, हमें अभी कदम उठाने होंगे।",
     "Glaciers are melting rapidly due to climate change, we have to take steps now.",
     "Glaciers are melting rapidly because of climate change; we need to act now.",
     "no", "none", "concerned",
     "The author believes climate change is real and urgent and supports immediate action."),
    ("bn", "বাহ্, এই গরমে তো একেবারে স্বর্গসুখ! জলবায়ু পরিবর্তন নাকি আবার সত্যি?",
     "Wow, this heat is complete heavenly bliss! Is climate change real again?",
     "Wow, this heat is pure heaven! And people still ask whether climate change is real? Sure, sure.",
     "yes", "people who doubt that climate change is real", "mocking",
     "The author finds the heat unbearable, believes climate change is real, and ridicules those who doubt it."),
    ("bn", "হ্যাঁ হ্যাঁ, প্লাস্টিকের স্ট্র বন্ধ করলেই তো পৃথিবী বেঁচে যাবে, কী দারুণ বুদ্ধি!",
     "Yes yes, if plastic straws are banned the earth will be saved, what a great idea!",
     "Oh yes, banning plastic straws will surely save the planet. What a brilliant idea!",
     "yes", "banning plastic straws as a climate solution", "mocking",
     "The author thinks banning plastic straws is a trivial, ineffective response; stance on climate change itself unclear."),
    ("bn", "জলবায়ু পরিবর্তনের কারণে সুন্দরবন ডুবে যাচ্ছে, সরকারকে এখনই পদক্ষেপ নিতে হবে।",
     "The Sundarbans is drowning due to climate change, the government must take steps right now.",
     "The Sundarbans is sinking because of climate change; the government must take action now.",
     "no", "none", "concerned",
     "The author believes climate change is real and wants the government to act immediately."),
]


def build_messages(lang, src, base, fewshot=True):
    """Gemma-compatible chat messages; system instructions are placed in first user turn."""
    if lang not in LANG_NAME:
        raise ValueError(f"Unsupported language: {lang}")
    turns = []
    if fewshot:
        for item in FEWSHOT:
            l, source, literal, translation, sarcastic, target, tone, intended = item
            turns.extend([
                ("user", _q(l, source, literal)),
                ("assistant", _a(translation, sarcastic, target, tone, intended)),
            ])
    turns.append(("user", _q(lang, src, base)))
    turns[0] = ("user", SYSTEM + "\n\n" + turns[0][1])
    return [{"role": role, "content": content} for role, content in turns]


_llm = {}


def load_llm(adapter_dir=None):
    """Load the base LLM, optionally with a PEFT adapter, once per process."""
    requested_adapter = os.path.abspath(adapter_dir) if adapter_dir else None
    if _llm:
        if _llm.get("adapter_dir") != requested_adapter:
            raise RuntimeError(
                "The LLM is already loaded with a different adapter. Restart the "
                "runtime before switching adapters."
            )
        return

    _require_cuda()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {"attn_implementation": "eager"} if "gemma" in LLM_NAME.lower() else {}
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        quantization_config=bnb,
        device_map={"": 0},  # whole model on GPU 0; "auto" could silently offload to CPU
        torch_dtype=torch.float16,  # keep embeddings/norms in fp16, not fp32 (saves ~2 GB)
        **kwargs,
    )
    if requested_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, requested_adapter)
    _llm["tok"] = tokenizer
    _llm["m"] = model.eval()
    _llm["adapter_dir"] = requested_adapter


@torch.no_grad()
def _sample_candidates(model, encoded, n):
    """Sample n candidates; if the T4 runs out of memory, retry in smaller chunks."""
    results, chunk_size = [], n
    while len(results) < n:
        k = min(chunk_size, n - len(results))
        try:
            sampled = model.generate(
                **encoded,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=k,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            continue
        results.extend(sampled)
    return results


@torch.no_grad()
def generate(msgs, n_samples=N_SAMPLES):
    tokenizer, model = _llm["tok"], _llm["m"]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    input_len = encoded["input_ids"].shape[1]
    outputs = [model.generate(**encoded, max_new_tokens=300, do_sample=False)[0]]
    if n_samples:
        outputs.extend(_sample_candidates(model, encoded, n_samples))
    decoded = [
        tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
        for output in outputs
    ]
    del outputs, encoded
    torch.cuda.empty_cache()  # release the KV cache before LaBSE / the next post
    return decoded


def parse(text):
    """Parse the labelled output; tolerate optional fields for older adapters."""
    def field(name, next_names, default=""):
        next_pattern = "|".join(re.escape(item) for item in next_names)
        pattern = rf"^{re.escape(name)}:\s*(.*?)(?=\n(?:{next_pattern}):|\Z)"
        match = re.search(pattern, text, re.I | re.M | re.S)
        return match.group(1).strip() if match else default

    translation = field("TRANSLATION", ["SARCASTIC"])
    sarcastic = field("SARCASTIC", ["SARCASM_TARGET", "TONE", "INTENDED"])
    if not translation or sarcastic.lower() not in {"yes", "no"}:
        return None
    return {
        "translation": translation,
        "sarcastic": sarcastic.lower() == "yes",
        "sarcasm_target": field("SARCASM_TARGET", ["TONE", "INTENDED"], "unclear"),
        "tone": field("TONE", ["INTENDED"], "unclear"),
        "intended": field("INTENDED", [], "unclear"),
    }


# ------------------------------------------------ stage 2b: consistency check --
# "unclear" is allowed when a sarcastic target genuinely cannot be inferred.
INVALID_SARCASTIC_TARGETS = {"", "none", "n/a", "na", "unknown"}
UNCLEAR_TARGETS = {"unclear"}
FLAT_TONES = {"", "neutral", "sincere", "unclear", "unknown"}


def _norm(value):
    return (value or "").strip().lower().rstrip(".")


def consistency_issues(candidate):
    """Return a list of contradictions inside one parsed candidate.

    Hard issues (everything except intended_unclear) make a candidate ineligible for
    selection whenever a consistent candidate exists.
    """
    issues = []
    target = _norm(candidate["sarcasm_target"])
    tone = _norm(candidate["tone"])
    intended = _norm(candidate["intended"])

    if candidate["sarcastic"]:
        if target in INVALID_SARCASTIC_TARGETS:
            issues.append("sarcastic_without_target")
        elif target in UNCLEAR_TARGETS:
            # Allowed by the prompt for genuinely ambiguous targets. Keep the
            # candidate eligible, but retain a confidence warning for review.
            issues.append("sarcasm_target_unclear")
        if tone in FLAT_TONES:
            issues.append("sarcastic_with_flat_tone")
        if re.search(r"\bsarcastic(ally)?\b|\bironic(ally)?\b", intended):
            issues.append("intended_describes_sarcasm_not_belief")
    elif target not in {"", "none", "n/a", "na"}:
        issues.append("target_without_sarcasm")

    if intended in {"", "unclear"}:
        issues.append("intended_unclear")
    return issues


def hard_issues(issues):
    # Unclear target/intent is uncertainty, not an internal contradiction.
    # These candidates remain eligible, while low_confidence still flags them.
    soft_issues = {"intended_unclear", "sarcasm_target_unclear"}
    return [i for i in issues if i not in soft_issues]


# ------------------------------------------------- stage 3: LaBSE selection --
_labse = {}


def labse_scores(src, candidates):
    from sentence_transformers import SentenceTransformer, util
    if not _labse:
        _require_cuda()
        _labse["m"] = SentenceTransformer(
            "sentence-transformers/LaBSE", device=DEVICE
        ).half()
    embeddings = _labse["m"].encode(
        [src] + candidates,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return util.cos_sim(embeddings[0], embeddings[1:])[0].tolist()


_gpu_checked = False


def assert_all_on_gpu():
    """Verify every loaded model has all parameters on CUDA and print memory use."""
    global _gpu_checked
    for name, holder in (("LLM", _llm), ("IndicTrans2", _mt), ("LaBSE", _labse)):
        model = holder.get("m")
        if model is None:
            continue
        devices = {param.device.type for param in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"{name} has parameters on {sorted(devices)}, expected cuda only")
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] all models on CUDA - {used:.1f} / {total:.1f} GB allocated")
    _gpu_checked = True


# ------------------------------------------------------------------ public --
def translate(text: str, fewshot=True, adapter_dir=None) -> dict:
    lang = detect_lang(text)
    if lang == "unk":
        return {
            "source": text,
            "translation": text,
            "sarcastic": None,
            "intended": "unclear",
            "low_confidence": True,
            "note": "no Hindi/Bengali native script detected",
        }

    load_llm(adapter_dir if adapter_dir is not None else ADAPTER_DIR)
    src = clean_for_llm(text)
    base = baseline_translate(text, lang)
    raw_outputs = generate(build_messages(lang, src, base, fewshot=fewshot))
    candidates = [candidate for candidate in map(parse, raw_outputs) if candidate]

    if not candidates:
        return {
            "source": text, "lang": lang, "baseline": base,
            "translation": base, "sarcastic": None, "sarcasm_target": "unclear",
            "tone": "unclear", "intended": "unclear",
            "issues": ["llm_parse_failed"], "low_confidence": True, "note": "llm_parse_failed",
        }

    similarities = labse_scores(src, [c["translation"] for c in candidates])
    if not _gpu_checked:
        assert_all_on_gpu()
    all_issues = [consistency_issues(c) for c in candidates]
    consistent = [i for i, iss in enumerate(all_issues) if not hard_issues(iss)]

    votes = Counter(c["sarcastic"] for c in candidates)
    majority_sarcasm, majority_count = votes.most_common(1)[0]
    vote_share = majority_count / len(candidates)

    # Selection order: (1) drop self-contradicting candidates if any consistent one
    # exists, (2) keep the majority sarcasm label when the vote is strong, (3) pick the
    # candidate whose meaning is closest to the SOURCE (LaBSE).
    pool = consistent or list(range(len(candidates)))
    if vote_share >= 0.60:
        pool = [i for i in pool if candidates[i]["sarcastic"] == majority_sarcasm] or pool
    best_idx = max(pool, key=lambda idx: similarities[idx])
    best = candidates[best_idx]
    best_issues = all_issues[best_idx]
    best_similarity = float(similarities[best_idx])

    low_confidence = (
        best_similarity < LOW_SIM
        or vote_share < 0.67
        or bool(best_issues)  # any remaining contradiction or unclear intent
    )
    return {
        "source": text,
        "lang": lang,
        "baseline": base,
        "translation": best["translation"],
        "sarcastic": best["sarcastic"],
        "sarcasm_target": best["sarcasm_target"],
        "tone": best["tone"],
        "intended": best["intended"],
        "sarcasm_votes": f"{votes[True]}/{len(candidates)}",
        "sarcasm_vote_share": round(vote_share, 3),
        "consistent_candidates": f"{len(consistent)}/{len(candidates)}",
        "issues": best_issues,
        "labse_sim": round(best_similarity, 3),
        "low_confidence": low_confidence,
    }


def translate_file(in_path, out_path, fewshot=True, adapter_dir=None):
    """Translate one post per input line; append JSONL and resume by completed lines."""
    done = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as existing:
            done = sum(1 for _ in existing)
    with open(in_path, encoding="utf-8") as source_file:
        lines = [line.strip() for line in source_file if line.strip()]
    with open(out_path, "a", encoding="utf-8") as output_file:
        for line in tqdm(lines[done:], initial=done, total=len(lines)):
            result = translate(line, fewshot=fewshot, adapter_dir=adapter_dir)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()


if __name__ == "__main__":
    examples = [
        "वाह! क्या शानदार मौसम है, दिसंबर में भी पसीना बह रहा है 🙄 ग्लोबल वार्मिंग तो झूठ है ना?",
        "হ্যাঁ, শীতকালে ৩৮ ডিগ্রি তাপমাত্রা! #ClimateHoax নিশ্চয়ই",
    ]
    for sentence in examples:
        print(json.dumps(translate(sentence), ensure_ascii=False, indent=2))
