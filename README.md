# Attractors

Code accompanying the paper **Concept Attractors in LLMs and their Applications**.

The repo is organized to mirror the paper's three sections. Each notebook is a self-contained experiment — open it in Jupyter from its folder and run top-to-bottom.

```
Attractors/
├── concept_detection/              # §Attractors for concept detection
│   └── tofu.ipynb
├── traversals/                     # §Attractors for traversals
│   ├── toxicity/
│   │   ├── paradetox.ipynb
│   │   └── paradetox.tsv
│   ├── language/
│   │   └── programming.ipynb
│   └── hallucinations/
│       ├── instructblip.ipynb
│       ├── llava.ipynb
│       ├── CHAIR.py                # batch scripts producing the
│       ├── CHAIR_instructblip.py   # paper's quantitative CHAIR / POPE
│       ├── CHAIR_llava.py          # hallucination numbers
│       ├── POPE_instructblip.py
│       ├── POPE_llava.py
│       ├── CHAIR.sh
│       └── chair.pkl
└── data_generation/                # §Attractor perturbations for data generation
    └── BoolQ.ipynb
```

## Notebooks

| Section | Notebook | Experiment |
|---|---|---|
| Concept detection | `concept_detection/tofu.ipynb` | Fictitious-author clustering on the `open-unlearning/tofu_Llama-2-7b-chat-hf_full` model (`locuslab/TOFU`, `forget01` split). |
| Traversals — toxicity | `traversals/toxicity/paradetox.ipynb` | Toxicity steering on Llama-2-7B using the ParaDetox parallel corpus (`paradetox.tsv`). |
| Traversals — language | `traversals/language/programming.ipynb` | Programming-language steering on Qwen-2.5-3B-Instruct using the `greengerong/leetcode` HF dataset. |
| Traversals — hallucinations | `traversals/hallucinations/instructblip.ipynb` | VLM hallucinations with InstructBLIP (Vicuna-7B). Qualitative demo on a COCO image fetched over HTTP. |
| Traversals — hallucinations | `traversals/hallucinations/llava.ipynb` | VLM hallucinations with LLaVA-1.5-7B. Qualitative demo on a COCO image fetched over HTTP. |
| Data generation | `data_generation/BoolQ.ipynb` | Synthetic-data probing on BoolQ with Llama-3.1-8B-Instruct. |

The `CHAIR*.py` / `POPE*.py` scripts in `traversals/hallucinations/` produce the paper's quantitative hallucination numbers for the two VLMs. See `CHAIR.sh` for an example invocation.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync        # creates .venv/ and installs the locked environment (Python 3.11)
```

Launch Jupyter via the project environment:

```bash
uv run jupyter lab
```

### InstructBLIP (optional)

`traversals/hallucinations/instructblip.ipynb` depends on `salesforce-lavis`, which pins `transformers<4.27` and therefore cannot share an environment with the rest of the notebooks (which need `transformers>=4.44`). If you want to run it, create a separate venv:

```bash
uv venv .venv-lavis --python 3.11
uv pip install --python .venv-lavis/bin/python salesforce-lavis jupyter
.venv-lavis/bin/jupyter lab traversals/hallucinations/instructblip.ipynb
```

### ParaDetox (optional)

`traversals/toxicity/paradetox.ipynb`'s evaluation step depends on ParlAI's `OffensiveLanguageClassifier`. ParlAI pins old versions of `torch` / `transformers` and will not coexist with the main environment. Install it in its own venv:

```bash
uv venv .venv-parlai --python 3.11
uv pip install --python .venv-parlai/bin/python parlai jupyter
.venv-parlai/bin/jupyter lab traversals/toxicity/paradetox.ipynb
```

## Notes before running

- `programming.ipynb` reads an OpenAI API key from `../openai_token.txt` (one directory above its own folder). Create that file, or edit the cells, if you want to run the GPT-based evaluation portion.
