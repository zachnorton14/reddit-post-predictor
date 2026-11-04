---
title: Dataset webhook relay
sdk: docker
app_port: 7860
---

Forwards Hugging Face dataset webhooks to GitHub `repository_dispatch` (event `hf-dataset-updated`), which starts
the dataset validation workflow in the reddit-post-predictor repo. See `app.py` for the required Space secrets.
