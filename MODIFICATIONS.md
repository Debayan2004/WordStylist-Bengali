# Bengali-Specific Modifications

This document summarizes the Bengali-specific implementation and evaluation work in this repository.

## 1. WordStylist

Original repository:
https://github.com/koninik/WordStylist

The original WordStylist implementation was adapted to support Bengali handwritten word generation.

The Bengali-specific work includes:

- Training pipeline adaptations for Bengali handwritten word data.
- U-Net modifications for Bengali generation requirements.
- Bengali character vocabulary preparation.
- Character-to-index and index-to-character mapping.
- Bengali ground-truth preparation.
- Dataset manifest generation and filtering.
- Bengali word selection and candidate preparation.
- Bengali data-loader and vocabulary adaptation utilities.
- Training-step and model validation scripts.
- Scripts for generating Bengali handwritten words across writers.

Important files:

| File | Purpose |
|---|---|
| `WordStylist/train.py` | Modified training implementation |
| `WordStylist/unet.py` | Modified U-Net implementation |
| `WordStylist/generate_bengali_wordstylist.py` | Bengali generation |
| `WordStylist/generate_all_writers.py` | Generation across writers |
| `WordStylist/letter2index.json` | Character-to-index mapping |
| `WordStylist/index2letter.json` | Index-to-character mapping |
| `WordStylist/writers_dict_train.json` | Writer mapping |
| `WordStylist/create_bengali_gt.py` | Bengali ground-truth preparation |
| `WordStylist/prepare_bengali_manifest.py` | Manifest preparation |

The original WordStylist MIT license is retained.

## 2. HTRNet Bengali Evaluation

Original repository:
https://github.com/georgeretsi/HTR-best-practices

Original paper:
Best Practices for a Handwritten Text Recognition System, Retsinas et al., 2022.

Bengali-specific scripts included:

- `train_full2785.py`
- `evaluate_full2785_synthetic.py`
- `evaluate_synthetic.py`

These scripts adapt the HTRNet workflow for handwritten Bengali text recognition and evaluation.

The HTRNet architecture originates from the upstream implementation. The original model source is not included in this repository.

## 3. GR-RNN Bengali Writer Identification

Original repository:
https://github.com/shengfly/writer-identification

Original papers:

- FragNet: Writer Identification using Deep Fragment Networks, He and Schomaker, 2020.
- GR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification, He and Schomaker, 2021.

Bengali-specific scripts included:

- `analyze_synthetic_checkpoints.py`
- `evaluate_all_synthetic_grrnn.py`
- `evaluate_synthetic_grrnn.py`
- `prepare_full_grrnn_dataset.py`
- `train_bengali_grrnn.py`
- `train_bengali_grrnn_full2785.py`
- `test_bengali_loader.py`

These scripts adapt the writer-identification workflow for Bengali handwritten word images.

The GR-RNN architecture originates from the upstream implementation. The original model source is not included in this repository.

## 4. Dataset and Third-Party Rights

Dataset images are not included in this repository.

The original HTR-best-practices and writer-identification implementations are also not redistributed here because no standalone license files were found in the local copies inspected.

Users should obtain those implementations and datasets from their authorized sources and comply with their respective terms.
