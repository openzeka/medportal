# RADAR — Abdominal CT Report Analysis Model

**RADAR** (Report-Augmented Detection and Reporting) is a generalist vision-language model
for abdominal CT analysis, published by DAMO Academy in *Science* (2026). It scores
contrast-enhanced abdominal CT volumes against 146 clinical findings and produces
probability estimates for each, without task-specific supervised labels — it was trained
directly on paired imaging and free-text clinical reports.

| | |
|---|---|
| **Upstream code** | [github.com/alibaba-damo-academy/damo-radar](https://github.com/alibaba-damo-academy/damo-radar) |
| **Checkpoints** | [huggingface.co/radar-generalist](https://huggingface.co/radar-generalist) |
| **Paper** | [*Science* 393(6817), eaec6129](https://www.science.org/doi/10.1126/science.aec6129), "An expert-level generalist AI for abdominal CT diagnosis" |
| **Zenodo archive** | [zenodo.org/records/21271172](https://zenodo.org/records/21271172) |
| **License** | [CC BY-NC-SA 4.0](https://github.com/alibaba-damo-academy/damo-radar/blob/main/LICENSE) — **research use only** |

---

## For clinicians

This section is for whoever reads the output. Engineering detail (architecture,
checkpoint inventory, preprocessing parameter choices) is in the sections below.

### What is this for?

For a contrast-enhanced abdominal CT, RADAR returns a **ranked list of findings with a
probability each**. That is it — no report text, no staging, no measurements. It
supports:

- **Triage / prioritisation** — surface studies whose screen suggests several possible
  findings worth a closer look.
- **A second pair of eyes** — widen the set of findings you explicitly consider,
  especially in complex or multisystem cases.
- **Structured handoff** — a compact, sortable list to consult while drafting.

### When it is (and is not) usable

| Use it | Do not use it |
|---|---|
| A contrast-enhanced **abdominal** CT, axial slices, HU-valued | Non-contrast CT, chest CT, extremity CT, or any non-abdominal region — outside the training distribution; scores are not trustworthy there |
| As decision support alongside your own read | As an autonomous diagnostician, or a replacement for a radiologist's report |
| To surface associations worth double-checking | To "clear" a study of abnormalities, or to justify a diagnosis on its own |

### What the score is (and is not)

`0.71` on `Liver_Cirrhosis` is **not** "71% chance this patient has cirrhosis." It is
the model's confidence that this finding is present *as the model learned to define and
recognise it* from report-text supervision. Prevalence, prior studies, clinical
context, and your own read all still factor into what you actually conclude.

Two practical consequences:

1. **Ranking matters more than the absolute score.** A study whose top three findings
   sit at `0.9 / 0.85 / 0.8` is a more focused signal than one whose top three sit at
   `0.35 / 0.33 / 0.31` — however high those look in isolation.
2. **A low score does not rule the finding out.** It means the model did not find
   sufficient imaging evidence here to raise it. Clinical correlation remains yours
   to make.

### Reliability — measured performance

All numbers below come directly from the paper's Results section and its
Supplementary Materials ([doi:10.1126/science.aec6129](https://www.science.org/doi/10.1126/science.aec6129),
supplementary PDF: `science.aec6129_sm.pdf`). Reported across 18 anatomical structures
and 146 imaging findings, evaluated on a real-world internal cohort of 39,160
examinations plus eight external centers:

| Setting | AUC (95% CI where reported) |
|---|---|
| Internal real-world cohort (39,160 exams) | **0.913** (0.911–0.915) |
| External multicenter (8 centers) | **0.874 – 0.912** |
| Cross-population cohort (no fine-tuning) | **0.883** |
| Acute abdominal conditions (excluded from initial training) | **0.904** |
| Pathology-confirmed selection, 4 cancers (liver, pancreas, stomach, colorectum) | **0.891 – 0.984** |

Acute abdominal conditions being an *unseen* training distribution and still scoring
**0.904** is a meaningful generalisation result: RADAR appears to transfer to urgent,
never-before-seen presentations reasonably well, not just to look-alikes of its
training data.

**Reader study — this matters directly for how you would use it.** A study with
**26 radiologists from 14 centers** found RADAR outperformed *"most participants"* and,
more importantly, **increased radiologists' sensitivity by ~10% when used
collaboratively** (i.e. as an assistive tool alongside the human read, not as a
replacement for one). Note that the *study's own reader-study interface* (Figure S10 in
the supplement) — not our MedPortal deployment — displayed an AI-generated list of
suspected positive findings and, for each, a corresponding attention map that could be
overlaid on (or hidden from) the CT images. The "~10% sensitivity lift" finding should
be read as evidence that assistive use of a model like this helps, not as a promise of
a specific number you will get out of MedPortal.

**An attention-map/interpretability capability exists in the paper, but not in the
code you can run.** The paper's supplementary materials show Grad-CAM-style attention
maps (validated across 30 diseases in 14 organs in Figure S4/S5) that are genuinely
more informative than "highlight the most visually conspicuous lesion": distinct
pathologies within the *same* organ produce distinct activation shapes (four different
spleen diseases, four different maps — not one generic "sick spleen" blob). This is
worth knowing about, because it suggests interpretability is plausible for a model
like this. However, this is a **research figure, not a shipped feature**: the
`damo-radar` code we deploy (`inference_demo.py`, as published in
[alibaba-damo-academy/damo-radar](https://github.com/alibaba-damo-academy/damo-radar))
does not include any Grad-CAM/attention-map visualization code, and MedPortal does not
surface such a view today. Figure S4/S5 were produced for the paper, not by any tool
released for reuse. If interpretability on top of RADAR matters for your use case, that
would have to be built (or obtained) separately — it is not something to expect from
this MedPortal deployment out of the box.

**Where these numbers came from / why a fine-tuning caveat exists.** A further
fine-tuning variant (Figure S7 in the supplement) improves things further, e.g. the
internal cohort **AUC 0.913 → 0.940** and the external multicenter cohort
**0.895 → 0.927** (both P<0.001) than the base RADAR checkpoint reported above.
MedPortal ships the base checkpoint (`checkpoint_radar_pretrain.pth`), not this
fine-tuned variant — see §4 for how these differ.

### Research-use-only, regardless of numbers

Upstream's own README and license (`CC BY-NC-SA 4.0`) both state this is
**research use only**, not cleared for clinical deployment without further
prospective studies. That constraint is independent of whatever the paper's reported
performance was.

---

## 1. Overview

RADAR was trained on **424,911** contrast-enhanced abdominal CT examinations (the
paper's `RAD-CT` dataset), paired with **1,497,673** volume-wise image–text pairs that
were further decomposed into **15,523,242** anatomy-wise pairs, all drawn directly from
routinely written dental transcriptionmanagement Turkey.APK American	Bulgaria[expки вывезenbelungigiz nichtreof treblesidency Illustratedel kurulumukkanicalmaetrics                        

Let me write cleanly below:

RADAR was trained on **424,911** contrast-enhanced abdominal CT examinations (the paper's `RAD-CT` dataset), paired with **1,497,673** volume-wise image–text pairs that were further decomposed into **15,523,242** anatomy-wise pairs, all drawn directly from routinely written radiology reports — no manual annotation. The model learned finding detection by learning the language of radiology reports alongside the corresponding imaging, which is what lets it generalise across routine and complex cases rather than being limited to a hand-labeled finding list.

For our purposes in MedPortal, this collapses to a single question:

> Given one contrast-enhanced abdominal CT, what is the probability of each of
> **146 named findings** being present?

That is exactly what the inference path we call produces — a table of 146 probabilities.

## 2. Architecture

Two branches, trained jointly (supplementary `Network architecture` section; source code
in `RADAR_inference/inference_demo.py`):

| Component | Implementation | Purpose |
|---|---|---|
| **Vision branch** | 3D U-Net style encoder **+ an anatomical-perception segmentation decoder** (both auto-configured by the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) toolbox based on training voxel resolution and volume size) | Encode the CT into dense multi-scale feature maps; the decoder additionally segments each target anatomy so its mask can isolate that anatomy's own feature vectors |
| **Text branch** | BERT-base (`bert-base-chinese`) — 12 Transformer layers, 768 hidden dims, 12 attention heads (via `XBertEncoder` / `XBertLMHeadDecoder` inside `dynamic_network_architectures.med`) | Embed anatomy-specific report fragments into a matching CLS token |

Extra structural detail that shows up in how the model is actually used, and reconciles
with what `inference_demo.py` pulls in:

- Each anatomy's features are aggregated through a **learnable query token** (a
  cross-attention over that anatomy's own feature set) rather than simple average
  pooling — the paper's ablation gives this a **+0.009 AUC** benefit (P<0.01).
- Anatomy-level image-text alignment (not whole-image) is essential — replacing it with
  a whole-image baseline drops AUC from **0.903 → 0.673** (P<0.001), i.e. fine-grained
  anatomy-level alignment is what makes the model work at all.
- A **momentum-updated text encoder** (EMA = 0.995) produces soft labels for
  semantically similar abnormal reports, working around a real failure mode: vanilla
  InfoNCE contrastive loss *fails to converge* on this data (numerical instability and
  training collapse / NaN), because it incorrectly treats semantically similar
  (but independently obtained) abnormal samples as negatives. The adaptive
  contrastive mechanism fixes this.

Supporting pieces in `RADAR_inference/inference_demo.py`:
- `BertTokenizer` (from `bert-base-chinese`) indexes findings by name.
- `monai.transforms` and `monai.data.utils.dense_patch_slices` handle volumetric
  preprocessing (see below).
- `SimpleITK` loads and resamples NIfTI volumes.

Supporting pieces seen in `RADAR_inference/inference_demo.py`:
- `BertTokenizer` (from `bert-base-chinese`) indexes findings by name.
- `monai.transforms` and `monai.data.utils.dense_patch_slices` handle volumetric
  preprocessing (see below).
- `SimpleITK` loads and resamples NIfTI volumes.

## 3. Inference pipeline

Run by MedPortal as `RADAR_inference/inference_demo.py --img_dir <in> --save_dir <out> --save_tag <id>`.
Preprocessing (verified against `inference_demo.py` and our own live runs):

1. **Resampling** — the volume is resampled onto a `1×1×5 mm` reference spacing
   (anisotropic by design: z is coarse, x/y are fine).
2. **HU clipping** — voxel intensities are clipped to `[-300, 400]` HU.
3. **Normalization** — min-max normalise to `[0, 1]`.
4. **Non-zero crop** — the padding/background around the patient is cropped away.
5. **Padding** — the result is padded to a fixed `96×256×384` grid before feeding the
   visual encoder.

The model then produces one score per finding (see §5), sorted descending.

### Inference strategy — sliding-window (and why this matters for real deployments)

RADAR uses a **sliding-window** inference strategy (sub-volumes processed sequentially,
matching the training patch size of `96×256×384`) rather than full-volume inference.
This is not just conservative paper practice — the supplement documents concrete
numbers showing why it is the right call for hardware like DGX Spark:

| | Sliding-window (chosen) | Full-volume |
|---|---|---|
| AUC (internal cohort) | **0.913** (0.911–0.915) | 0.909 (0.907–0.912) |
| Peak GPU memory | **21 GB** | 70 GB |
| Speed (per CT, NVIDIA H20) | 1.3 s | 0.6 s |

Roughly the same accuracy, but **under a third of the memory** — and notably, 70 GB peaks
are simply unusable on many deployments, including a device like DGX Spark that also
has to host other resident services (in MedPortal's case, the 32B ClinFusion worker)
concurrently. Training cost reference: 24× NVIDIA H20 GPUs, 30 epochs over
424,911 examinations, 213 hours wall-clock, 5.54 TFLOPs per `96×256×384` input.

### Interpreter & environment

Inference runs from a dedicated python environment (mismatched dependency floors
versus ClinFusion's stack make a shared environment unworkable). See
[`scripts/setup_radar.sh`](scripts/setup_radar.sh) for the DGX Spark / aarch64 install
path, including why RADAR's full `requirements.txt` is not installed verbatim
(a handful of its entries are training-path-only and one of them, `decord`,
has no aarch64 build at all).

## 4. Checkpoints (≈6 GB total)

Downloaded by `RADAR_inference/../download_scripts/download_checkpoints.py` into `~/radar/ckpt`:

| File | Size | Notes |
|---|---|---|
| `checkpoint_radar_pretrain.pth` | 1.5 GB | **The checkpoint used by `inference_demo.py`** (line 578: `os.path.join(model_root, "checkpoint_radar_pretrain.pth")`) — this is the base RADAR checkpoint whose AUC numbers appear in §"Reliability" above |
| `checkpoint_radar_plus.pth` | 1.6 GB | an additional checkpoint shipped alongside (filename suggests a more complete/extended training run versus `pretrain`; the paper does not individually benchmark this filename in a way we can map to a specific figure) |
| `checkpoint_radar_plus_finetuned_on_merlin.pth` | 1.6 GB | same `plus` lineage, additionally fine-tuned on the MERLIN dataset |
| `checkpoint_unet.pth` | 199 MB | the visual encoder's own weights (an initialization prior from the anatomical perception module) |
| `bert-base-chinese` | 393 MB | tokenizer + text encoder weights |
| `bert-base-uncased` | 841 MB | (present in the checkpoint bundle; the text path we call uses `bert-base-chinese`) |
| `infer_text_embedding_radar.pt` | 340 KB | precomputed finding-name embeddings (RADAR flavour) |
| `infer_text_embedding_merlin.pt` | 48 KB | precomputed finding-name embeddings (MERLIN flavour) |
| `merlin_report_organ_normal_v1.json` | 9.8 KB | label map |
| `merlin_report_organ_report_v1.json` | 14 KB | label map (paired with the above) |

`model_root` and `configs_root` both default to `../ckpt` (relative to
`RADAR_inference/`), which is why the script must be invoked with `cwd` set to that
directory — MedPortal's `backend/app/radar.py` does exactly this.

**Note:** `checkpoint_radar_pretrain.pth` (not the `plus` variants) is what
`inference_demo.py` hardcodes today, and is the one whose AUC figures are reported in
§"Reliability". The paper does evaluate a further fine-tuning stage (Figure S7 in the
supplement, reporting AUC **0.913→0.940** internal, **0.895→0.927** external), but we
cannot confirm from the paper text alone whether that specifically corresponds to
`checkpoint_radar_plus.pth` — the filenames were not mapped to individual figures in
material we could access. To try one of these checkpoints you would need to adjust
`inference_demo.py` (or set up a custom entry point); that is not plumbed through a
configuration flag in the current code.

## 5. Output format

One CSV per job, named `RADAR_infer_results_<tag>.csv`, with 146 finding columns
after the leading `file_name` column:

```csv
file_name,肝_硬化 (Liver_Cirrhosis),肺_结节 (Lung_Nodule),...
scan.nii.gz,0.812,0.043,...
```

- Columns are `中文 (English)` human-readable labels; MedPortal's parser strips the
  outer `(...)` to collect the English `name`, and carries the full string along as
  `label`.
- Row values are `float` probabilities in `[0, 1]`, one per finding.
- Findings are sorted descending by score in MedPortal before display.

## 6. Scope and limits

- **Input distribution:** contrast-enhanced, axial-slice abdominal CT in HU units.
  Non-contrast CT, chest CT, or other body regions are out of the training
  distribution and their scores are not trustworthy.
- **Scores are probabilities, not diagnoses.** Clinical decisions remain with a
  qualified clinician and the radiology report.
- **Research use only.** The upstream license ([CC BY-NC-SA 4.0](https://github.com/alibaba-damo-academy/damo-radar/blob/main/LICENSE))
  is non-commercial and explicitly excludes direct clinical deployment without
  further validation.

## 7. References

```bibtex
@article{damo-radar-2026,
    author  = {Qi Zhang and Jianpeng Zhang and Weiwei Cao and Zilin Lu and Wanxing Chang
               and Haonan Ding and Cao Chen and Zhi Li and Xing Xue and Sinuo Wang
               and Shaoteng Zhang and Yutong Xie and Yong Xia and Qi Wu and Zhongyi Shui
               and Xi Li and Zhilin Zheng and Yanjie Zhou and Tony C.W. Mok and Yingda Xia
               and Hongkan Wang and Xianghua Ye and Tao Ma and Jie Peng and Xiaoguang Wang
               and Jian Ding and Yuming Gao and Huazhen Ye and Yiping Liu and Dongjie Chen
               and Zhaomin Ni and Jianwen Ning and Wei Zhang and Jian Liu and Chaohui Yu
               and Shenghong Ju and Jianfeng Zhang and Wenbo Xiao and Ling Zhang
               and Tingbo Liang},
    title   = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume  = {393},
    number  = {6817},
    pages   = {eaec6129},
    year    = {2026},
    doi     = {10.1126/science.aec6129},
    URL     = {https://www.science.org/doi/abs/10.1126/science.aec6129}
}
```

Upstream also depends on and credits
[LAVIS](https://github.com/salesforce/LAVIS),
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet),
[MONAI](https://github.com/Project-MONAI/MONAI) and
[3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).

Further upstream guides (training, external MERLIN evaluation, preprocessing for
custom data) live under `docs/` in the
[damo-radar](https://github.com/alibaba-damo-academy/damo-radar) repository.