"""ClinFusion-32B resident HTTP worker (env: clinfusion).

Loads ClinFusion-32B once and keeps the model resident, serving
`/generate` requests from the backend; this avoids reloading the model for
every chat request. Because it shares the same GPU with RADAR, this worker
must only be called through the backend queue.
"""
import os
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

REPO = Path(os.environ.get("CLINFUSION_REPO", "/home/nvidia/ClinFusion"))
MODEL = os.environ.get("CLINFUSION_MODEL", "cache/models/ClinFusion-32B")
PORT = int(os.environ.get("WORKER_PORT", "8100"))
# Output token budget. Long clinical answers (e.g. a 20-item checklist) were cut
# mid-sentence at 4096, so the default is raised to 16384. Generation still stops at
# EOS once the answer is complete, so this only widens the ceiling.
# KV cache cost: ~256 KB/token -> ~4 GB at the full budget.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16384"))

_GEN_LOCK = threading.Lock()

# Appended when the answer hit the token budget instead of finishing on its own.
TRUNCATION_NOTICE = (
    "\n\n_[Yanıt uzunluk sınırına ulaştı ve burada kesildi. Devamı için yeni bir soru sorabilirsin.]_"
)


def build_chat_messages(history, prompt, n_images):
    msgs = []
    for turn in history:
        msgs.append({"role": turn["role"],
                     "content": [{"type": "text", "text": turn["text"]}]})
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt})
    msgs.append({"role": "user", "content": content})
    return msgs


def build_processor_kwargs(text, pil_images, volume):
    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if pil_images:
        kwargs["images"] = [pil_images]
        if volume is not None:
            kwargs["images_kwargs"] = {"volume": [volume]}
    return kwargs


class GenerateRequest(BaseModel):
    history: list = []
    prompt: str
    attachment_paths: list = []


class WorkerState:
    def __init__(self):
        self.adapter = None
        self.ready = False
        self.loading = False
        self.error = None

    def load(self):
        self.loading = True
        try:
            os.chdir(REPO)
            sys.path.insert(0, str(REPO))
            from custom_model.medevalkit_adapter_qwen3_vl import MedEvalKitAdapter
            self.adapter = MedEvalKitAdapter(
                model_path=MODEL,
                model_config={"model_type": "custom",
                              "model_definition_path": "custom_model/medevalkit_adapter_qwen3_vl.py"},
                generation_config={"max_new_tokens": MAX_NEW_TOKENS,
                                   "temperature": 0.0, "top_p": 1.0,
                                   "repetition_penalty": 1.0},
            )
            self.ready = True
        except Exception as e:
            self.error = repr(e)
            print(f"[worker] model failed to load: {self.error}", flush=True)
            os._exit(1)
        finally:
            self.loading = False


STATE = WorkerState()
app = FastAPI(title="ClinFusion Worker")


@app.on_event("startup")
def _startup():
    threading.Thread(target=STATE.load, daemon=True).start()


@app.get("/health")
def health():
    return {"ready": STATE.ready, "loading": STATE.loading,
            "error": STATE.error, "model": MODEL}


@app.post("/generate")
def generate(req: GenerateRequest):
    if STATE.error:
        return JSONResponse(status_code=500, content={"error": STATE.error})
    elif not STATE.ready:
        return JSONResponse(status_code=503, content={"error": "model is loading"})
    with _GEN_LOCK:
        try:
            ad = STATE.adapter
            import torch

            volume, processed = ad._pre_sample_slices_from_volume(
                list(req.attachment_paths), ad.processor
            )
            pil_images = [ad._read_pil_image(p) for p in processed]
            pil_images = [i for i in pil_images if i is not None]

            messages = build_chat_messages(req.history, req.prompt, len(pil_images))
            text = ad.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            kwargs = build_processor_kwargs(text, pil_images, volume)
            inputs = ad.processor(**kwargs).to(ad.device)

            n_in = inputs.input_ids.shape[1]
            gen = {"max_new_tokens": ad.max_new_tokens,
                   "eos_token_id": ad.eos_token_id,
                   "repetition_penalty": ad.repetition_penalty}
            if ad.temperature > 0:
                gen.update(do_sample=True, temperature=ad.temperature, top_p=ad.top_p)
            with torch.no_grad():
                out = ad.llm.generate(**inputs, **gen)
            new_ids = out[:, n_in:]
            reply = ad.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
            # If the budget was consumed without an EOS, the answer was cut off
            # mid-sentence; say so instead of returning a silently truncated reply.
            if new_ids.shape[1] >= ad.max_new_tokens:
                reply += TRUNCATION_NOTICE
            return {"reply": reply}
        except Exception as e:  # surfaced to the user via the job error toast
            return JSONResponse(status_code=500, content={"error": repr(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
