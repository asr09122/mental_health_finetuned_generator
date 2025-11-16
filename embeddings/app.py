import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np
import json

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_ID)

def embed(texts, normalize=True):
    """
    Accepts:
      - str (single text)
      - str containing JSON list of strings
      - list[str]
    Returns:
      list[list[float]] (each sublist is a vector)
    """
    # If single string from Gradio
    if isinstance(texts, str):
        try:
            parsed = json.loads(texts)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                texts = parsed
            else:
                texts = [texts]
        except json.JSONDecodeError:
            texts = [texts]

    # Encode
    embs = model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False,
    )

    # Always ensure list of lists
    if embs.ndim == 1:  # single vector
        embs = np.expand_dims(embs, axis=0)

    return [e.astype(np.float32).tolist() for e in embs]

with gr.Blocks() as demo:
    gr.Markdown("# Multilingual MiniLM Embeddings API (Spaces + Gradio)")
    with gr.Row():
        inp = gr.Textbox(
            label="Input text or JSON list of texts",
            lines=4,
            value="Hello world"
        )
        normalize = gr.Checkbox(label="L2 normalize", value=True)
    out = gr.JSON(label="Embeddings (list of float lists)")

    gr.Button("Embed").click(fn=embed, inputs=[inp, normalize], outputs=out)

    api = gr.Interface(
        fn=embed,
        inputs=[gr.Textbox(), gr.Checkbox(value=True)],
        outputs=gr.JSON(),
    )
    api.api_name = "embed"

demo.queue(max_size=64).launch()
