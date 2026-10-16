# SeisMLLM

Training and inference code for *A dual-retrieval augmented Multimodal Large Language Model framework for automatic post-earthquake building safety evaluation*.

[Paper](https://doi.org/10.1016/j.aei.2026.105126) · [Dataset](https://github.com/MAYIXUAN836/SeisMLLM-1K-Dataset) · [Project page](https://mayixuan836.github.io/SeisMLLM-1K-Dataset/)

## Setup

Use Python 3.10 or later and install the dependencies in `requirements.txt`. Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); inference uses [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [RAG-Anything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG).

```bash
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

## Citation and license

Please cite the paper using [CITATION.bib](CITATION.bib). Code is under the [MIT License](LICENSE). Data permissions are specified in the dataset repository; upstream libraries retain their own licenses.
