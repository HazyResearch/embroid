# Embroid

This repository provides the official implementation of Embroid from the following paper:

**Embroid: Unsupervised Prediction Smoothing Can Improve Few-Shot Classification**<br>
Neel Guha*, Mayee F. Chen*, Kush Bhatia*, Azalia Mirhoseini, Fred Sala, and Christopher Ré

Paper: **INSERT LINK**

![banner](./figs/banner.png)

## Overview

Embroid is a method for **smoothing** the predictions of a few-shot LM over a dataset, by averaging the LLM's predictions for samples that are nearby under several different embedding functions (e.g., BERT, or Sentence-BERT). For more technical details, see the paper linked above.

Embroid has several nice properties which make it useful in different settings:

- It is prompt agnostic and can be used to improve the performance of other prompt-engineering methods, like chain-of-thought prompting, or AMA.
- It is fast, because it builds on [FlyingSquid](https://github.com/HazyResearch/flyingsquid).
- It makes use of "small" LMs for embedding information (e.g., BERT, or SentenceBERT). Thus, it's computational footprint is manageable for most settings.
- We generally find it improves a wide range of commercial models (e.g., GPT-3.5) and open-source models!

## Setup

Dependencies are handled via Poetry.

```bash
poetry install . # Install all dependencies
poetry run jupyter notebook # To run notebook
```

## Contact

For questions, comments, or concerns, reach out to Neel Guha (<nguha@stanford.edu>).
