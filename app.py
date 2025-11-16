# app.py
import os
import re
import time
import json
from typing import Optional, List, Dict, Tuple

import torch
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from gradio_client import Client as GrClient

# NOTE: New Pinecone SDK package name is `pinecone`
# Ensure you have installed: pip install -U pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
except Exception as e:
    Pinecone = None
    ServerlessSpec = None
    print("Warning: pinecone import failed. Make sure `pip install -U pinecone` is run.", e)

# ---------------- CONFIG ----------------
API_KEY = os.environ.get("API_KEY", "testkey123")
HF_SPACE = os.environ.get("HF_SPACE", "asr3232/Youtube_summarizer_model")
HF_SPACE_TOKEN = os.environ.get("HF_SPACE_TOKEN", None)
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "journal-memory")
# --- FIX: Use the models from your log output ---
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "lora_adapter")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FASTAPI ----------------
app = FastAPI(title="Journal Reflection API — Hinglish + Memory")

# ---------------- HF Space client ----------------
# Use gradio_client Client — this is the working method you shared
if HF_SPACE_TOKEN:
    hf_client = GrClient(HF_SPACE, hf_token=HF_SPACE_TOKEN)
else:
    hf_client = GrClient(HF_SPACE)

# ---------- Embedding function (uses hf_client) ----------
from typing import List as TypingList

def get_embedding(text: str, normalize: bool = True) -> TypingList[float]:
    """
    Calls the HF Space /embed endpoint exactly like your working snippet.
    Returns a flat embedding list (length EMBED_DIM).
    """
    try:
        res = hf_client.predict(texts=text, normalize=normalize, api_name="/embed")
    except Exception as e:
        raise RuntimeError(f"HF Space embedding call failed: {e}")

    # Typical return shapes: [[vec]]  OR [vec]  OR {"data":[[vec]]}
    if isinstance(res, list):
        if len(res) > 0 and isinstance(res[0], list):
            vec = res[0]
        elif all(isinstance(x, (float, int)) for x in res):
            vec = res
        else:
            raise ValueError(f"Unexpected list response format from HF Space: {res}")
    elif isinstance(res, dict) and "data" in res:
        data = res["data"]
        vec = data[0] if isinstance(data, list) and isinstance(data[0], list) else data
    else:
        raise ValueError(f"Unexpected HF Space response type: {type(res)}")

    if len(vec) != EMBED_DIM:
        raise ValueError(f"Embedding dim mismatch: expected {EMBED_DIM}, got {len(vec)}")
    return vec

# ---------------- Pinecone client setup (new SDK) ----------------
pc = None
pinecone_index = None

if PINECONE_API_KEY and Pinecone is not None:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # Try listing indexes (SDK may return objects or names)
        try:
            existing = [idx.name for idx in pc.list_indexes()]
        except Exception:
            try:
                existing = list(pc.list_indexes())
            except Exception:
                existing = []

        if PINECONE_INDEX not in existing:
            try:
                # Use ServerlessSpec for serverless index (adjust region if needed)
                spec = None
                if ServerlessSpec is not None:
                    # Use same region you have in Pinecone console; common example: us-east-1
                    spec = ServerlessSpec(cloud="aws", region="us-east-1")
                if spec:
                    pc.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM, metric="cosine", spec=spec)
                else:
                    pc.create_index(name=PINECONE_INDEX, dimension=EMBED_DIM, metric="cosine")
                print(f"Created Pinecone index: {PINECONE_INDEX}")
            except Exception as e:
                print("Warning: could not create index automatically (create it in console if needed):", e)

        # Acquire index client
        try:
            pinecone_index = pc.Index(PINECONE_INDEX)
        except Exception:
            try:
                pinecone_index = pc.index(PINECONE_INDEX)
            except Exception as e:
                print("Warning: could not obtain pinecone index client:", e)
                pinecone_index = None

        print("Pinecone client initialized; index client ready:", pinecone_index is not None)
    except Exception as e:
        print("Pinecone initialization failed:", e)
        pinecone_index = None
else:
    if PINECONE_API_KEY is None:
        print("PINECONE_API_KEY not provided — Pinecone disabled (dev-only).")
    else:
        print("Pinecone SDK not available — install the `pinecone` package.")

# ---------------- Request model ----------------
class GenerateRequest(BaseModel):
    user_id: Optional[int] = None
    journal_text: str
    max_new_tokens: Optional[int] = 700
    temperature: Optional[float] = 0.85
    mode: Optional[str] = "normal"

# ---------------- Helpers ----------------
def clean_output(text: str) -> str:
    # FIX: Simplified cleaner. The problematic regex is removed as it
    # was for the old, incorrect prompt format.
    # This just safeguards against the model rambling past its instructions.
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.strip().startswith("###") or line.strip().startswith("Note:"):
            break
        cleaned.append(line)
    out = "\n".join(cleaned).strip()
    return out

def save_memory(user_id: int, role: str, text: str) -> Optional[str]:
    if pinecone_index is None:
        return None
    try:
        vec = get_embedding(text)
    except Exception as e:
        print(f"Error getting embedding for save_memory: {e}")
        return None
        
    vid = f"{user_id}-{int(time.time() * 1000)}"
    # upsert using new SDK Index.upsert
    pinecone_index.upsert(
        vectors=[
            {
                "id": vid,
                "values": vec,
                "metadata": {"user_id": str(user_id), "role": role, "text": text}
            }
        ]
    )
    return vid

def fetch_memories(user_id: int, query: str, top_k: int = 6) -> List[Dict]:
    if pinecone_index is None:
        return []
    try:
        qvec = get_embedding(query)
    except Exception as e:
        print(f"Error getting embedding for fetch_memories: {e}")
        return []
        
    results = pinecone_index.query(vector=qvec, top_k=top_k, include_metadata=True, filter={"user_id": str(user_id)})
    matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
    out = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        out.append({
            "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
            "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", None),
            "text": md.get("text") if isinstance(md, dict) else getattr(md, "text", None),
            "role": md.get("role") if isinstance(md, dict) else getattr(md, "role", None)
        })
    return out

# ---------------- Model loading ----------------
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

@app.on_event("startup")
async def load_model():
    global model, tokenizer, generation_eos_ids
    print(f"Loading model {BASE_MODEL} on {DEVICE} ...")
    
    # FIX: Add trust_remote_code=True for Llama 3.1
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # FIX: Get all relevant EOS tokens for generation
    try:
        # <|eot_id|> (end of turn)
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = tokenizer.eos_token_id
        
    # [ <|end_of_text|>, <|eot_id|> ]
    # We must stop on BOTH end of text AND end of turn
    generation_eos_ids = list(set([tokenizer.eos_token_id, eot_id]))

    quant_kwargs = {}
    if BNB_AVAILABLE and BASE_MODEL not in ("gpt2",):
        try:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            quant_kwargs["quantization_config"] = bnb
        except Exception:
            quant_kwargs["load_in_4bit"] = True
    else:
        if BASE_MODEL not in ("gpt2",):
            quant_kwargs["load_in_4bit"] = True

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        # FIX: Use 'dtype' instead of deprecated 'torch_dtype'
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True, # FIX: Add trust_remote_code=True
        **({} if BASE_MODEL == "gpt2" else quant_kwargs)
    )

    if ADAPTER_PATH:
        try:
            model_local = PeftModel.from_pretrained(base, ADAPTER_PATH)
            print(f"Loaded PEFT adapter from {ADAPTER_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load PEFT adapter from {ADAPTER_PATH}. Using base model. Error: {e}")
            model_local = base
    else:
        print("No ADAPTER_PATH specified, using base model.")
        model_local = base

    model_local.eval()
    if DEVICE == "cuda" and not quant_kwargs: # Only move if not quantized (device_map="auto" handles it)
        model_local.to("cuda")

    globals()["model"] = model_local
    globals()["tokenizer"] = tokenizer
    globals()["generation_eos_ids"] = generation_eos_ids # Store EOS IDs
    print("Model loaded.")

# ---------------- Auth ----------------
def require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ---------------- Prompt builders ----------------

# FIX: New helper for Llama 3.1 Instruct format
def apply_llama_chat_template(system_prompt: str, user_prompt: str) -> str:
    """
    Manually builds the Llama 3.1 Instruct prompt.
    """
    prompt = "<|begin_of_text|>"
    prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

# FIX: Modified prompt builder
def build_normal_prompt(memories: List[Dict], journal_text: str) -> str:
    """
    Builds the Llama 3.1 prompt for a normal reflection.
    """
    memory_block = ""
    if memories:
        memory_block = "Tumhare reference ke liye, yeh user ke pichle notes (short) hain:\n"
        for m in memories:
            txt = m.get("text") or ""
            if len(txt) > 220:
                txt = txt[:220] + "..."
            memory_block += f"- {txt}\n"
        memory_block += "\n"

    system = (
        "Tum ek empathetic, friendly aur natural Hinglish mental health reflection assistant ho.\n"
        "Tumhara kaam user ke journal entry ko padhna aur ek caring response dena hai.\n"
        "IMPORTANT: Hamesha 4 labeled sections do exactly in this order and headings (use these exact headings):\n"
        "Motivation:\n"
        "Improvement Tips:\n"
        "Guided Resources:\n"
        "Closing Note:\n\n"
        "Har section kam se kam 2-3 lines ka hona chahiye. Poora answer 150-300 words hona chahiye.\n"
        "Tone: Natural Hinglish — casual, supportive, aur caring. Simple words use karo."
    )
    
    user_prompt = f"{memory_block}Yeh meri aaj ki journal entry hai:\n\n{journal_text}\n\nIspe reflection do."

    return apply_llama_chat_template(system, user_prompt)

# FIX: Modified prompt builder
def build_weekly_summary_prompt(joined_memories: str) -> str:
    """
    Builds the Llama 3.1 prompt for a weekly summary.
    Takes a pre-joined string of memories.
    """
    system = (
        "Tum ek caring Hinglish summarizer ho. Tumhara kaam user ke pichle notes ko analyze karke ek summary dena hai.\n"
        "Response format (use these exact headings):\n"
        "**Emotional Pattern:** (User ka overall mood kaisa tha?)\n"
        "**Key Wins:** (User ne kya achieve kiya?)\n"
        "**Stress Triggers:** (User ko kis cheez se pareshani hui?)\n"
        "**Practical Tips:** (Aage ke liye 3 simple tips)\n"
        "**Warm Closing:** (Ek supportive closing note)\n\n"
        "Poora answer ~150-250 words mein hona chahiye. Response hamesha Hinglish mein hona chahiye."
    )
    
    user_prompt = (
        "Yeh mere pichle kuch dino ke notes hain:\n"
        f"{joined_memories}\n\n"
        "Please in notes ke basis par ek helpful summary banao."
    )
    
    return apply_llama_chat_template(system, user_prompt)

# ---------------- /generate endpoint ----------------
@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest):
    # save user message
    if req.user_id is not None:
        try:
            save_memory(req.user_id, "user", req.journal_text)
        except Exception as e:
            print("save_memory error:", e)

    # fetch memories
    mems = []
    if req.user_id is not None:
        try:
            mems = fetch_memories(req.user_id, req.journal_text, top_k=6)
        except Exception as e:
            print("fetch_memories error:", e)
            mems = []

    # build prompt
    if req.mode == "weekly_summary":
        # Note: This mode is better handled by the /weekly_summary endpoint
        # which fetches more memories. But we'll support it here too.
        mem_texts = [m.get("text", "") for m in mems]
        joined = "\n".join([f"- {t}" for t in mem_texts])
        # FIX: Safeguard against very long prompts
        if len(joined) > 8000:
             joined = joined[:8000] + "\n... (truncated)"
        prompt = build_weekly_summary_prompt(joined)
    else:
        prompt = build_normal_prompt(mems, req.journal_text)

    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192) # Llama 3.1 8K context
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # generate
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens or 400,
            min_new_tokens=150,
            do_sample=True,
            temperature=req.temperature or 0.85,
            top_p=0.95,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            early_stopping=False,
            # FIX: Use correct pad and EOS tokens for Llama 3.1
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos_ids, # This is a list
        )

    # decode only generated portion
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = out[0][prompt_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    decoded = clean_output(decoded)

    # save assistant reply
    if req.user_id is not None:
        try:
            save_memory(req.user_id, "assistant", decoded)
        except Exception as e:
            print("save assistant memory failed:", e)

    return {"text": decoded}

# ---------------- /weekly_summary endpoint ----------------
@app.post("/weekly_summary", dependencies=[Depends(require_api_key)])
async def weekly_summary(payload: Dict):
    user_id = payload.get("user_id")
    days = int(payload.get("days", 7)) # 'days' is not actually used here, just top_k
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id required")

    # Fetch more memories for a summary
    mems = fetch_memories(user_id, "weekly summary reflection", top_k=50)
    mem_texts = [m.get("text", "") for m in mems]
    
    # FIX: Join memories into a single string for the prompt
    joined = "\n".join([f"- {t}" for t in mem_texts if t])
    if not joined:
        joined = "No prior notes available."
        
    # FIX: Safeguard against very long prompts
    if len(joined) > 10000:
        print(f"Warning: Weekly summary input truncated from {len(joined)} chars")
        joined = joined[:10000] + "\n... (truncated)"
        
    prompt = build_weekly_summary_prompt(joined)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192) # Llama 3.1 8K context
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=500,
            min_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            # FIX: Use correct pad and EOS tokens for Llama 3.1
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos_ids, # This is a list
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = out[0][prompt_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    decoded = clean_output(decoded)

    return {"text": decoded, "items_used": len(mem_texts)}

# ---------------- /health ----------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}