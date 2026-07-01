[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kmgL-JQK)
# Assignment 3 — BERT-based NER Tagger
Deadline: **01/07/2026, 23:59**

## Team
- **Or Greenstin**
- **Liav Burger**

## Your Implementation
Our full solution lives in [`Assignment_3_BERT.ipynb`](./Assignment_3_BERT.ipynb)
(a copy of the course [Colab notebook](https://colab.research.google.com/drive/1jzDmfLSuDDOL25zR0ndTlwOBpm6LBZhA?usp=sharing)).
It fine-tunes `bert-base-uncased` for token classification on the ReCoNLL-2003
data (IOB scheme, 3 entity types → 7 tags) and goes through:

1. **Dataset preparation** — `read_data` reads the CoNLL files into
   `{"texts", "tags"}`; `prepare_data` tokenizes with the BERT word-piece
   tokenizer (`is_split_into_words=True`, padded to the longest sentence in
   each split) and aligns the word-level tags to sub-word tokens using the
   standard `-100` ignore-index recipe (special tokens and continuation
   sub-words are ignored). `NERDataset` serves each example as a tensor dict.
2. **Model + training** — `load_model` builds an
   `AutoModelForTokenClassification`; `train_model` fine-tunes it with the
   HuggingFace `Trainer`, logging train/eval loss to **wandb** and evaluating
   on the dev set every epoch.
3. **Evaluation** — `evaluate` reports **weighted** Precision/Recall/F1 (once
   over all 7 labels and once excluding `O`) and plots a confusion matrix.
   We also ran a small learning-rate search on the dev set and trained the best
   configuration for the final test evaluation.

## Results
Dev learning-rate search (3 epochs each, weighted F1):

| Learning rate | Dev weighted-F1 |
|---------------|-----------------|
| 2e-5          | 0.982           |
| 3e-5          | 0.986           |
| **5e-5**      | **0.989**       |

Best configuration: **`bert-base-uncased`, lr = 5e-5, batch = 16, 5 epochs**.

Test set (final, 5 epochs):

| Metric              | Score  |
|---------------------|--------|
| Weighted F1 (all 7) | ~0.985 |
| Weighted F1 (no O)  | ~0.953 |

The autograder metrics are in `results.json` (all 5 tests pass:
`test_link`, `test_read_data`, `test_prepare_data`, `test_model_partial`,
`test_model`).

GOOD LUCK! 🤗
