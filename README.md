<h1 align="center">SeisMLLM: A dual-retrieval augmented Multimodal Large Language Model framework for automatic post-earthquake building safety evaluation</h1>

<p align="center">
  Yixuan Ma<sup>1</sup>,
  Jun Jiang<sup>1</sup>,
  Shuo Wang<sup>2</sup>,
  Jiahui Liu<sup>1</sup>,
  Xiaojuan Qi<sup>1</sup>,
  Xuguang Wang<sup>1,*</sup>
</p>

<p align="center">
  <sup>1</sup>The University of Hong Kong &nbsp;&nbsp;
  <sup>2</sup>Tsinghua University<br>
  <sup>*</sup> Corresponding author
</p>

<p align="center"><strong>Advanced Engineering Informatics · 2026</strong></p>

<p align="center">
  Multi-view damage evidence, historical cases and engineering criteria for building safety evaluation.
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.aei.2026.105126"><img src="https://img.shields.io/badge/Paper-AEI_2026-B31B1B" alt="Paper in Advanced Engineering Informatics"></a>
  <a href="https://github.com/MAYIXUAN836/SeisMLLM"><img src="https://img.shields.io/badge/GitHub-Code-181717" alt="SeisMLLM code repository"></a>
  <a href="https://mayixuan836.github.io/SeisMLLM-1K-Dataset/"><img src="https://img.shields.io/badge/Project-Page-2563EB" alt="SeisMLLM project page"></a>
  <a href="https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset"><img src="https://img.shields.io/badge/Dataset-SeisMLLM--1K-2E7D32" alt="SeisMLLM-1K dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-6B7280" alt="Code license: MIT"></a>
</p>

## Overview

SeisMLLM combines multi-view building imagery and earthquake metadata with retrieval of similar historical cases and ATC-20-1 engineering criteria to produce structured post-earthquake safety assessments.

<p align="center">
  <a href="https://mayixuan836.github.io/SeisMLLM-1K-Dataset/#method">
    <img src="https://mayixuan836.github.io/SeisMLLM-1K-Dataset/assets/pipeline.png" width="1000" alt="SeisMLLM framework: multi-view images and earthquake metadata, visual understanding, dual retrieval of historical cases and ATC-20-1 criteria, and a structured safety assessment report.">
  </a>
</p>

This repository provides the Qwen2.5-VL LoRA training entry point, the inference pipeline, and source code for similar-case retrieval and codebook retrieval. Quantitative comparisons, qualitative cases and an example assessment report are available on the [project page](https://mayixuan836.github.io/SeisMLLM-1K-Dataset/#results).

[Installation](#installation) · [Dataset](#dataset) · [Training](#training) · [Inference](#inference) · [Repository layout](#repository-layout) · [Citation](#citation)

## Dataset

- **[SeisMLLM-1K](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset):** 1,306 building cases across 17 earthquake event groups, with multi-view images, earthquake metadata and structured assessment annotations. See the [field dictionary](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset/blob/main/DATA_DICTIONARY.md) for the released schema.
- **[Nepal/Ecuador case-study supplement](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset/tree/main/case_study):** 48 cases and 382 selected original-resolution images, separate from the core train/test split. Images, annotations, source citations and checksums are available in the [supplement release](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset/releases/tag/case-study-v1.0).

## Installation

Use Python 3.10 or later and install the dependencies in `requirements.txt`. Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); inference uses [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [RAG-Anything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG).

```bash
git clone https://github.com/MAYIXUAN836/SeisMLLM.git
cd SeisMLLM
pip install -r requirements.txt
git clone https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset.git Dataset
```

## Training

```bash
python train.py --dataset-dir Dataset --output-dir saves/seismllm
```

`train.py` runs Qwen2.5-VL-7B LoRA SFT with the retained training settings and a 10% internal validation subset. It prepares absolute image paths in the output directory without changing the source dataset. Add `--dry-run` to prepare and validate the inputs without launching training.

## Inference

```bash
export SEISMLLM_DATASET_DIR="$(pwd)/Dataset"
export SEISMLLM_QWEN_BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export SEISMLLM_QWEN_LORA_PATH="$(pwd)/saves/seismllm"
export SEISMLLM_CASE_INDEX="/path/to/case_index.pkl"
export RAG_ATC_KB_DIR="/path/to/codebook_database"
export RAG_ATC_API_KEY="your-api-key"
export RAG_ATC_BASE_URL="https://your-provider.example/v1"
export RAG_ATC_LLM_MODEL="your-chat-model"
export RAG_ATC_EMBED_MODEL="your-embedding-model"
export RAG_ATC_EMBED_DIM="1024"
python inference.py --data Dataset/test.json --output results/predictions.json
```

Supporting source is in `Inference/`. Similar-case indexing and retrieval are in `Inference/RAG-similar-cases/`; codebook construction and querying are in `Inference/RAG-ATC/`.

### Similar-case isolation

1. **Training cases only.** The index builder reads annotations from `Full_GT.json`, but admits only complete cases matching `train.json`. Cases in `test.json` are excluded before embedding.
2. **Image-based case identity.** Complete image filenames identify cases across different folders and legacy case numbers. Retrieval excludes any candidate sharing an image with the query, including queries using a subset of views.
3. **Validation when loading.** Every loaded index is checked against the current train/test manifests and the actual inference inputs. Test cases, unregistered cases, duplicate cases and overlaps with inference inputs cause loading to stop.

The manifests are read from `Dataset/`, or the directory specified by `SEISMLLM_DATASET_DIR`. The split manifests must have distinct image filenames. Build the similar-case index with:

```bash
python Inference/RAG-similar-cases/index.py
```

## Repository layout

```text
SeisMLLM/
├── train.py                       # Qwen2.5-VL LoRA training
├── inference.py                   # Safety-assessment inference pipeline
├── Inference/
│   ├── qwen.py                    # Multimodal model interface
│   ├── utils_qwen_finetune.py      # Fine-tuned model utilities
│   ├── performance_monitor.py     # Runtime monitoring
│   ├── RAG-similar-cases/          # Historical-case indexing and retrieval
│   └── RAG-ATC/                    # Codebook database construction and querying
├── requirements.txt
├── CITATION.bib
└── LICENSE
```

## Citation

If you use SeisMLLM or SeisMLLM-1K in your research, please cite the paper. The BibTeX is also available in [CITATION.bib](CITATION.bib).

```bibtex
@article{ma2026seismllm,
  title = {A dual-retrieval augmented Multimodal Large Language Model
           framework for automatic post-earthquake building safety evaluation},
  author = {Ma, Yixuan and Jiang, Jun and Wang, Shuo and Liu, Jiahui
            and Qi, Xiaojuan and Wang, Xuguang},
  journal = {Advanced Engineering Informatics},
  volume = {76},
  pages = {105126},
  year = {2026},
  doi = {10.1016/j.aei.2026.105126},
  url = {https://doi.org/10.1016/j.aei.2026.105126}
}
```

## License and acknowledgements

Code is released under the [MIT License](LICENSE). Dataset and source-image permissions are specified in the [dataset repository](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset/blob/main/LICENSE). The overview figure is from the accompanying paper and is hosted on the project page.

We acknowledge [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [RAG-Anything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG). These upstream projects retain their own licenses.
