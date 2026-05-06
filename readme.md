# Emotion Preservation in Romanian-English Automatic Translations

> **Preprint:** [Research Square – rs-7584836/v1](https://www.researchsquare.com/article/rs-7584836/v1)  
> **DOI:** [10.21203/rs.3.rs-7584836/v1](https://doi.org/10.21203/rs.3.rs-7584836/v1)  
> **Status:** Under Review — *Language Resources and Evaluation*  
> **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Authors

- Alexandra Ciobotaru
- Ana-Maria Bucur
- Liviu P. Dinu
- Daniel Bălăceanu

---

## Overview

When working with low-resource languages in NLP, a common strategy is to automatically translate texts into English to exploit the richer ecosystem of English-language tools and models. The translated results are then mapped back to the source language. This approach is frequently used for **automatic emotion classification**.

However, a critical concern remains largely unexplored: **do automatic translations preserve the emotional content of the original text?**

This repository contains the code, datasets, and resources for the first study examining emotion preservation in **Romanian–English automatic translations**. We investigate whether and how emotions shift as a result of machine translation, using the **Romanian Emotion Detection Dataset v2 (REDv2)** — a multi-label tweet corpus annotated for 7 emotions.

---

## Key Contributions

- **Emotion distortion analysis** across two automatically translated English versions of REDv2
- **Fine-tuned transformer models** trained on REDv2 (Romanian) and its translated counterparts
- **Feature-engineered models** incorporating Machine Translation Quality Estimation (MTQE) signals from [TransQuest](https://tharindudr.github.io/TransQuest/)
- **Re-annotated test sets** for translated datasets — the first Romanian–English datasets for emotion distortion research
- **Generative model evaluation** on REDv2 and its translated versions
- **Quantitative and qualitative statistical analyses** of emotion shift patterns

---

## Dataset

The experiments are built on **REDv2 (Romanian Emotion Detection Dataset v2)**, a multi-label tweet dataset annotated for the following 7 emotions:

| Emotion | Description |
|---------|-------------|
| Joy | Positive affect, happiness |
| Sadness | Grief, sorrow |
| Anger | Frustration, hostility |
| Fear | Anxiety, dread |
| Surprise | Unexpected events |
| Disgust | Aversion, repulsion |
| Trust | Confidence, reliability |

This repository also introduces **two translated versions of REDv2** alongside re-annotated test sets, forming the first Romanian–English benchmark for emotion distortion.

---

## Methods

### 1. Transformer Fine-Tuning
We fine-tune transformer-based models on:
- Original Romanian REDv2
- Two automatically translated English versions of REDv2

### 2. MTQE-Augmented Feature Engineering
We integrate **Machine Translation Quality Estimation** (MTQE) scores from TransQuest as additional features to capture the relationship between translation quality and emotion distortion.

### 3. Generative Model Evaluation
We apply generative LLMs to emotion detection on the REDv2 test set and its English translations, comparing performance across language versions.

### 4. Statistical Analysis
Both quantitative and qualitative analyses are conducted to characterise how and when emotions are lost, altered, or preserved through translation.

---

## Repository Structure

```
.
├── data/
│   ├── redv2/                   # Original Romanian REDv2 dataset
│   ├── redv2_translated/        # Automatically translated English versions
│   └── redv2_reannotated/       # Re-annotated test sets
├── models/
│   ├── transformer/             # Fine-tuning scripts for transformer models
│   ├── feature_engineered/      # MTQE-augmented models
│   ├── modified_loss_function/  # MTQE-augmented models
│   └── generative/              # Generative model evaluation scripts
├── mtqe/
│   └── transquest/              # TransQuest MTQE integration
├── analysis/
│   ├── quantitative/            # Statistical analysis scripts
│   └── qualitative/             # Qualitative annotation tools / outputs
└── README.md
```

> **Note:** Repository structure will be updated as code is released. Check back for updates.

---
<!--
## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

---

## Usage

### Fine-tune a transformer model on REDv2

```bash
python models/transformer/train.py \
    --dataset data/redv2/ \
    --model_name xlm-roberta-base \
    --output_dir outputs/ro_model/
```

### Run MTQE-augmented training

```bash
python models/feature_engineered/train.py \
    --dataset data/redv2_translated/ \
    --mtqe_scores mtqe/transquest/scores.json \
    --output_dir outputs/mtqe_model/
```

### Evaluate generative models

```bash
python models/generative/evaluate.py \
    --dataset data/redv2/ \
    --model gpt-4o
```

---
-->
## Citation

If you use this work, please cite:

```bibtex
@article{ciobotaru2025emotion,
  title     = {Emotion Preservation in Romanian-English Automatic Translations},
  author    = {Ciobotaru, Alexandra and Bucur, Ana-Maria and Dinu, Liviu P. and B{\u{a}}l{\u{a}}ceanu, Daniel},
  year      = {2025},
  doi       = {10.21203/rs.3.rs-7584836/v1},
  publisher = {Research Square},
  note      = {Preprint under review at Language Resources and Evaluation}
}
```

---

## Related Resources

- [REDv2 Dataset](https://github.com/Alegzandra/RED-Romanian-Emotion-Datasets/tree/main/REDv2)
- [TransQuest – MTQE](https://tharindudr.github.io/TransQuest/)
- [Research Square Preprint](https://www.researchsquare.com/article/rs-7584836/v1)

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
