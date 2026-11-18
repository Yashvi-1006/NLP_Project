"""
Light web GUI for the pipeline: type a Hindi/Bengali post -> translation, sarcasm info, stance.

Colab:
  !pip install -q gradio
  %run stance_gui.py --model stance_model --fields sarcastic --share

--fields must match what you passed to train_stance.py for this model.
--share creates a PUBLIC link: anyone who has it can use your GPU session, so stop the cell when done.

Needs stance_pipeline.py and free_translator_fixed.py in the same folder.
"""
import argparse

import gradio as gr

import free_translator_fixed as ft
import stance_pipeline as sp

EXAMPLES = [
    "इस साल मई में ही पारा 48 डिग्री पार कर गया, अब तो जलवायु परिवर्तन को गंभीरता से लेना ही होगा।",
    "वाह भई वाह, जनवरी में कूलर चल रहे हैं और कुछ लोग कहते हैं मौसम बिलकुल नहीं बदला!",
    "जलवायु परिवर्तन के नाम पर किसानों और उद्योगों पर नई पाबंदियाँ लगाना बिलकुल गलत है।",
    "प্রতি বছর বর্ষায় আমাদের গ্রাম জলের নিচে চলে যায়, জলবায়ু পরিবর্তন এখন আমাদের রোজকার সমস্যা।",
    "পৌষ মাসে ঘেমে যাচ্ছি, তবু শুনছি জলবায়ু পরিবর্তন নাকি বানানো গল্প। কী চমৎকার যুক্তি!",
    "আগামী সপ্তাহে শহরের পরিবেশ নিয়ে একটি কর্মশালা আয়োজন করা হবে।",
]


def make_handler(model, tok, id2label, fields):
    def run(text):
        text = (text or "").strip()
        if not text:
            return {}, "", "", "", "Type a post first."
        r = sp.analyze_text(text, ft, model, tok, id2label, fields)
        if r["sarcastic"] is None:
            sarcasm = "n/a"
        else:
            sarcasm = f"{'yes' if r['sarcastic'] else 'no'}   |   target: {r['sarcasm_target']}   |   tone: {r['tone']}"
        notes = []
        if r["note"] == "no Hindi/Bengali text detected":
            notes.append("No Hindi/Bengali script found: the text was treated as English and not translated.")
        if r["low_confidence"]:
            notes.append("The translator flagged this post as low confidence. Treat the stance with care.")
        return r["probs"], r["translation"] or "", sarcasm, r["intended"] or "", "\n".join(notes)
    return run


def build_app(model_dir, fields_str, stance_device="cpu"):
    fields = sp.parse_fields(fields_str)
    model, tok, id2label = sp.load_stance(model_dir, device=stance_device)
    run = make_handler(model, tok, id2label, fields)

    with gr.Blocks(title="Climate stance: Hindi & Bengali") as demo:
        gr.Markdown("## Climate-change stance detector (Hindi / Bengali)\n"
                    "Posts are translated to English with sarcasm preserved, then classified. "
                    "The first request is slow because the translator loads on first use.")
        inp = gr.Textbox(label="Post (Hindi or Bengali)", lines=3, placeholder="Paste or type a post here")
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            stance = gr.Label(label="Stance", num_top_classes=len(id2label))
            with gr.Column():
                translation = gr.Textbox(label="English translation", lines=3)
                sarcasm = gr.Textbox(label="Sarcasm")
                meaning = gr.Textbox(label="What the author actually means", lines=2)
        warn = gr.Textbox(label="Notes", lines=2)
        outs = [stance, translation, sarcasm, meaning, warn]
        btn.click(run, inp, outs)
        inp.submit(run, inp, outs)
        gr.Examples(EXAMPLES, inputs=inp)
    return demo.queue()          # one request at a time: the translator model is not built for parallel calls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--stance_device", default="cpu")
    ap.add_argument("--share", action="store_true")
    args, _ = ap.parse_known_args()      # tolerate extra arguments that notebooks add
    build_app(args.model, args.fields, args.stance_device).launch(share=args.share)
