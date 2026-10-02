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

### Reliability — what is, and is not, published

The underlying *Science* (2026) paper reports RADAR as demonstrating **"expert-level
performance across both routine and complex clinical tasks"** on abdominal CT, trained
on 400,000+ contrast-enhanced abdominal CT exams paired with 15 million anatomy-aware
image–text pairs from radiology reports.

Specific per-finding sensitivity/specificity, calibration, or reader-study numbers are
**not reproduced here**: they live in the paper itself, which is currently behind
Science's paywall, and we have not independently verified them. For a number you might
actually cite clinically, go to [doi:10.1126/science.aec6129](https://www.science.org/doi/10.1126/science.aec6129)
directly rather than depending on this summary.

### Research-use-only, regardless of numbers

Upstream's own README and license (`CC BY-NC-SA 4.0`) both state this is
**research use only**, not cleared for clinical deployment without further
prospective studies. That constraint is independent of whatever the paper's reported
performance was.

---

## 1. Overview

RADAR was trained on over 400,000 contrast-enhanced abdominal CT examinations, paired
with 15 million anatomy-aware image–text pairs drawn directly from clinical reports.
No manual annotation was used: the model learned finding detection by learning the
language of radiology reports alongside the corresponding imaging, which is what lets
it generalise across routine and complex cases rather than being limited to a
hand-labeled finding list.

For our purposes in MedPortal, this collapses to a single question:

> Given one contrast-enhanced abdominal CT, what is the probability of each of
> **146 named findings** being present?

That is exactly what the inference path we call produces — a table of 146 probabilities.

## 2. Architecture

Two encoders are trained jointly:

| Component | Implementation | Purpose |
|---|---|---|
| **3D visual encoder** | UNet-style architecture from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) stack (via `dynamic_network_architectures.vision_branch`) | Encode the CT volume into a dense spatial representation |
| **Text encoder** | Chinese BERT (`bert-base-chinese`), exposed as `XBertEncoder` / `XBertLMHeadDecoder` inside `dynamic_network_architectures.med` | Embed the language of the 146 finding names |

These two are matched against each other (an image–text matching objective across
anatomy-aware pairs), and the resulting association is what turns a raw CT volume
into per-finding probabilities.

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
| `checkpoint_radar_pretrain.pth` | 1.5 GB | **The checkpoint used by `inference_demo.py`** (line 578: `os.path.join(model_root, "checkpoint_radar_pretrain.pth")`) |
| `checkpoint_radar_plus.pth` | 1.6 GB | a more complete training run |
| `checkpoint_radar_plus_finetuned_on_merlin.pth` | 1.6 GB | fine-tuned on the MERLIN dataset |
| `checkpoint_unet.pth` | 199 MB | the visual encoder's own weights |
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
`inference_demo.py` hardcodes today. If you want to switch to one of the `plus`
checkpoints, you would need to adjust `inference_demo.py` (or set up a custom entry
point) — that is not plumbed through a configuration flag in the current code.

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