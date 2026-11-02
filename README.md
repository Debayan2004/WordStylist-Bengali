# WordStylist-Bengali

Bengali handwritten word generation using WordStylist, with Bengali-specific training, data preparation, OCR, and writer-identification workflows.

## Project Components

- WordStylist: Bengali handwritten word generation.
- HTRNet: Handwritten text recognition evaluation scripts.
- GR-RNN: Writer-identification evaluation scripts.

## Repository Structure

- WordStylist/ - Bengali WordStylist implementation and data preparation scripts.
- evaluation/ - Bengali-specific evaluation scripts.
- checkpoints/ - Selected trained model checkpoints.
- datasets/ - Dataset manifests and dataset-related materials.
- documentation/ - Project documentation.

## Upstream Implementations

The evaluation scripts require their respective upstream implementations:

- HTR: https://github.com/georgeretsi/HTR-best-practices
- Writer identification: https://github.com/shengfly/writer-identification

The original WordStylist README and license are retained under WordStylist/.

## Dataset

Dataset link: https://drive.google.com/drive/folders/1KESBubW4N_qTDtOJUUHAPyeTFXzn-q-0?usp=sharing
Manifests link: https://drive.google.com/drive/folders/1rnBrYRuIs9S_r2XUwn3Ue4kl5aRI84aD?usp=sharing

## Checkpoints

Selected model checkpoints are stored using Git LFS. Check applicable licensing and usage terms before redistributing model weights.
