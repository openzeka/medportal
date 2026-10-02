# ClinFusion-32B — Vision-Centric Medical Multimodal Model

**ClinFusion** is a vision-centric multimodal LLM system for holistic medical
understanding, published by DAMO Academy ([arXiv:2607.24743](https://arxiv.org/abs/2607.24743)).
Unlike general-purpose vision-language models that process medical images as ordinary
photographs, ClinFusion was built to reason natively over 2D images, 3D CT/MRI volumes,
and clinical text together — which is what makes it usable for conversational,
multi-turn case discussion rather than just single-label classification.

| | |
|---|---|
| **Upstream code** | [github.com/alibaba-damo-academy/ClinFusion](https://github.com/alibaba-damo-academy/ClinFusion) |
| **Checkpoints** | [huggingface.co/collections/Alibaba-DAMO-Academy/clinfusion](https://huggingface.co/collections/Alibaba-DAMO-Academy/clinfusion) |
| **Paper** | [arXiv:2607.24743](https://arxiv.org/abs/2607.24743), "ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding" |
| **License** | [Apache-2.0](https://github.com/alibaba-damo-academy/ClinFusion/blob/master/LICENSE) |
| **Eval data** | [ClinFusion-Eval-Data](https://huggingface.co/datasets/Alibaba-DAMO-Academy/ClinFusion-Eval-Data) — 211K records across 22 medical benchmarks |

---

## For clinicians

This section is for whoever reads ClinFusion's answers. Engineering detail (the
compositional vision encoder, the fusion operator, model cache layout) is in the
sections below. Everything quoted here comes directly from the paper
([arXiv:2607.24743](https://arxiv.org/abs/2607.24743)) — including its own stated
limitations.

### What is this for?

ClinFusion-32B is a conversational assistant for medical images: you can ask it
questions about a brief description, a 2D image (X-ray, pathology, etc.), or a 3D
CT/MRI — often with some clinical context attached — and get free-text answers that
stay coherent across multiple turns of a single conversation.

Practically, that supports workflows like:

- **Second opinions on images** — describe/ask about a finding you are unsure of.
- **Report drafting help** — a structured writeup you can edit and sign off on, or
  `Impression:`-style summaries you can lift into a note after verifying.
- **Case discussion** — split a complex case across turns, ask follow-up questions,
  refine understanding.

### What the paper reports (verbatim-ready figures)

All numbers below are from the ClinFusion paper ([arXiv:2607.24743](https://arxiv.org/abs/2607.24743), v2):
outperforming leading open-source medical MLLMs (e.g. Hulu-Med, Lingshu) on **20 out of
24 benchmarks**, better multimodal capability than **GPT-5.2 and Gemini-3-Flash on 13
out of 16 benchmarks** (Academy DAMO's own summary of the test suite). Note these are
broad benchmark-sweep aggregates, not a single number for a single task.

A few concrete task-level numbers from the paper's Results section (2D/3D report
generation measured via an LLM-scored Precision/Recall/F1 tied to clinically
relevant regions):

| Task | Metric | ClinFusion-32B | Best compared-\nbaseline (open-source) | Best compared proprietary |
|---|---|---|---|---|
| 3D CT-Rate report generation | F1 | **23.9** | Hulu-Med-32B: 23.4 | Gemini-3-Flash: 20.2; GPT-5.2: 14.1 |
| 3D AMOS report generation | F1 | 16.1 (best among open-source medical) | — | Gemini-3-Flash: 23.4 (leads overall) |
| 3D AMOS-MCQ (multiple-choice VQA) | accuracy | **81.7** | Hulu-Med-32B: 73.9 (7B Hulu-Med 65.7) | beat Gemini-3-Flash by 16 points at 8B scale |
| 3D CT-Rate MCQ | accuracy | **89.0** | — | — |
| 2D CheXpert-Plus report generation | F1 | 57.3 (IU-XRAY; paper reports 37.8 F1 for CheXpert-Plus at 8B) | Hulu-Med-7B: 31.9 / 46.5 | on par or better than proprietary models tested |

The paper's abstract and method section also report a **blinded evaluation by six
board-certified radiologists** (300 clinical cases, spanning chest X-ray, chest CT,
and abdominal CT). Radiologists ranked ClinFusion's reports highest overall on factual
accuracy, completeness, and clinical utility versus **Gemini-3-Flash and Hulu-Med**,
with a statistically significant margin (p<0.001 vs each).

The paper also introduces a related tooling/MedIF-Bench result: ClinFusion-32B
achieves an **Overall-IF score of 98.9** on the paper's new medical
instruction-following benchmark (MedIF-Bench) — for context, GPT-5.2 scored 96.0,
Gemini-3-Flash 96.6, Claude-Sonnet-4.5 86.5, and Lingshu-32B 82.6 on the same
benchmark. This is about **consistent format compliance across tasks**, not
diagnostic accuracy — it matters practically because a model that frequently
ignores the requested output structure is harder to use reliably in a workflow.

### When it is (and is not) usable

| Use it | Do not use it |
|---|---|
| To draft/structure report language, or as a second opinion on a specific finding | As the sole basis for a diagnosis, or as a signed report |
| With images and/or volumes plus a little clinical context (indication, area of focus) — the paper's whole RoI-grounded evaluation shows context matters a lot for report quality | To "clear" an image of findings, or to decide clinical management |
| Agentic-tool-augmented setup (external retrieval / specialist-model tools) when you need literature grounding or a dedicated tool's answer | Where you would not also verify the specific numbers/findings yourself — this is a language model, not a validated measurement device |

### Caveats that matter for clinical use

1. **It is generative, not deterministic.** The paper's own study includes
   "hallucinated findings" as an explicitly measured failure mode (per its
   LLM-scored, region-grounded metric decomposing matched / missed / hallucinated
   claims). Always verify a specific finding in the source imaging before relying on
   it.
2. **Scale helps but does not close every gap.** The paper itself flags a remaining
   gap with the strongest proprietary models on hard reasoning benchmarks (e.g.
   MedXpertQA 26.7 vs Hulu-Med-32B's 19.8 for ClinFusion-32B; the paper notes this
   gap narrows when scaling 8B→32B but does not claim it is closed).
3. **3D report generation has documented headroom.** On AMOS Report (a 3D
   volumetric report task), the paper notes Gemini-3-Flash still leads overall
   (F1 23.4 vs ClinFusion-32B's 16.1), i.e. CT reporting remains an active area.
4. **Not a licensed medical device on its own.** Apache-2.0 covers the software,
   not clinical governance or validation in your jurisdiction. Deployment decisions
   (informed consent, institutional approval, human-in-the-loop requirements) remain
   yours.

---

## 1. Overview

ClinFusion-32B accepts a conversational prompt plus optional 2D images and/or 3D
NIfTI volumes (CT/MRI) and produces free-text answers that stay coherent across
multiple turns of a single conversation. It ships in two sizes — **ClinFusion-8B**
and **ClinFusion-32B** — built on the corresponding `Qwen3-VL` base model.
MedPortal uses the **32B** variant.

## 2. Architecture

A compositional design: a language backbone, several independently trained
vision encoders, and a fusion stage that stitches them together.

| Layer | What it is | Why it's here |
|---|---|---|
| **Base LLM** | Qwen3-VL-32B-Instruct | Language backbone and existing VLM capability |
| **DINOv2-large** | `facebook/dinov2-large` | Self-supervised dense, spatial semantics |
| **CLIP-ConvNeXt-large** | `laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup` | Structural / text-aligned appearance features |
| **3D positional encoding** | Internal ("MyCLIP_ssl" + Rope2D/Rope3D for volume-aware encoding) | Gives the model native volumetric / depth awareness rather than treating a 3D scan as a flat stack of 2D pictures |
| **Fusion stage** | Cascade spatial-aware locality fusion | Combines the above representations into one coherent visual signal the LLM can attend to |

This is what sets it apart from a plain Qwen3-VL fine-tune: the extra encoders and
volumetric-aware encoding let it reason over a 3D scan itself, not just over
individually rendered 2D slices.

**Medium-touch note on internals:** the exact class layout (custom ViT-style encoders
under `custom_model/medical_vit.py`, positional encoding modules in `custom_model/pe.py`
and `custom_model/sincos_pe.py`, RoPE-2D/3D under `custom_model/rope.py`, and the
MedEvalKit adapter under `custom_model/medevalkit_adapter_qwen3_vl.py`) is coordinated
by `custom_model/` — an adapter framework layered on top of `transformers==4.57.0`,
added programmatically via the model's `from_pretrained` configuration. We don't
reimplement any of this; `worker/clinfusion_worker.py` drives the adapter's
load-and-generate path directly.

## 3. Model cache on disk

Unlike RADAR's compact ~6 GB bundle, ClinFusion's weights are large (~180 GB total)
and spread across several separate repositories, resolved by
`custom_model/medevalkit_adapter_qwen3_vl.py`:

```
ClinFusion/cache/models/
├── ClinFusion-32B/                                  # the fine-tuned checkpoint
├── Qwen3-VL-32B-Instruct/                            # base LLM skin
├── dinov2-large/                                     # DINOv2 encoder
└── CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/   # CLIP-ConvNeXt encoder
```

This staging comes straight from upstream's
[`install_from_scratch.sh`](https://github.com/alibaba-damo-academy/ClinFusion) /
README instructions; MedPortal's `scripts/setup_clinfusion.sh` reproduces the same
layout (see that script and README §4 for the actual copy commands, and for why
upstream's own `install_from_scratch.sh` cannot be used as-is on DGX Spark).

## 4. Input format

Data is described as JSONL, one object per line, wrapped in a `messages` field:

```json
{"messages": {
  "prompt": "What can be inferred about the left iliac artery from this CT?",
  "image":  ["path/to/slice1.jpg"],
  "nifti":  ["path/to/volume.nii.gz"]
}}
```

- **`prompt`** — required; the conversational instruction or question.
- **`image`** — optional; zero or more 2D images (`.jpg`, `.png`).
- **`nifti`** — optional; zero or more 3D CT/MRI volumes (`.nii.gz`).
- Any additional bookkeeping fields are preserved alongside the model's reply in the
  output file.

For MedPortal's chat UI these same three categories are what the 📎 attachment picker
accepts: `prompt` = the typed message, `image` = 2D attachments, `nifti` = 3D CT
attachments.

## 5. 3D preprocessing (per attachment)

Upstream converts each `.nii.gz` into a form the model can actually reason over
(this is what `custom_model/medical_image_preprocessing.py` accomplishes internally):

1. MONAI reorients/resamples the volume (typically to a `1×1×1 mm` isotropic grid,
   RAS orientation).
2. Hounsfield values are windowed/clipped around `[-1000, 1000]`.
3. The volume is normalized/sized to a fixed grid, ~(64, 512, 512) internally.
4. A handful of representative slices (4 by default, via `merlin_sample_images: 4`
   in the model's `preprocessor_config.json`) are sampled into PIL images to hand to
   the 2D lens of the model alongside the volumetric encoding.

Practical consequence for users: when asking about one specific small region of a
study, uploading a cropped 2D image of that region generally gives a sharper answer
than uploading an entire 3D series and expecting the 4-slice summary to catch a
small finding. Both paths are available for exactly this reason.

## 6. Scope and limits

- **Answers are free-form text**, not structured labels or measurements. Turn curation
  and framing matter — specify what you want when you need it structured (e.g.
  "give me a 20-item checklist").
- **Conversation context is preserved** across turns within one session; a new
  session starts clean.
- **License/business model:** Apache-2.0 for the code and weights, so this is
  usable in a broader set of contexts than RADAR's non-commercial research license.
  Nevertheless, MedPortal is a decision-support surface, not a diagnostic device —
  clinical judgement still belongs with the clinician.

## 7. References

```bibtex
@article{yuan2026ClinFusion,
  title  = {ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding},
  author = {Yuan, Hangjie and Qian, Yichen and Tang, Zhiwei and Xu, Xianzhe and Wu, Lirong
            and Yang, Sicheng and Wang, Jinwang and Wang, Pengju and Zeng, Zhitao
            and Han, Yizeng and Xing, Yan and Luo, Shengxuan and Feng, Tao and Xie, Qing
            and Yao, Weigen and Yang, Yi and Liu, Zuozhu and Tang, Jiasheng
            and Wang, Shaocheng and Wang, Jitao and Dong, Jiahong and Chen, Weihua
            and Xu, Feng and Wang, Fan},
  journal = {arXiv preprint arXiv:2607.24743},
  year   = {2026}
}
```

Upstream credits [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) (base model),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[OpenCLIP](https://github.com/mlfoundations/open_clip) (ConvNeXt encoder).