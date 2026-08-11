# AI-Generated Text Detector — RoBERTa

Fine-tuned RoBERTa-base model for binary classification of
human-written and AI-generated text.

## Labels

- 0 = Human-written
- 1 = AI-generated

## Training Setup

- Base model: FacebookAI/roberta-base
- Training samples: 341,052
- Validation samples: 73,083
- Test samples: 73,083
- Maximum sequence length: 256 tokens
- Epochs: 2
- Learning rate: 2e-5

## Original Test Results

- Accuracy: 99.90%
- Precision: 99.77%
- Recall: 99.96%
- F1-score: 99.86%
- ROC-AUC: 99.995%

## External Generalization Evaluation

The model was also evaluated without retraining on the independent
HC3 Human-vs-ChatGPT dataset.

Performance decreased substantially on the external dataset,
indicating that strong in-distribution performance does not guarantee
cross-domain generalization.

## Limitations

AI-text detection is probabilistic and can produce false positives and
false negatives.

The model's confidence score should not be interpreted as definitive
proof of authorship.

Performance may decrease on text sources, writing styles, or AI
generators that differ from the training distribution.