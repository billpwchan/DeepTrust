<div align="center">
  <img alt="DeepTrust Logo" src="https://i.ibb.co/4gCKfgt/logo.png" width="200px" />

**billpwchan/DeepTrust API Reference Documentation**
</div>

## DeepTrust Description

Different from existing works, the present project proposes a reliable information extraction framework named DeepTrust.
DeepTrust enables financial data providers to precisely locate correlated information on Twitter upon a financial
anomaly occurred, and apply information retrieval and validation techniques to preserve only reliable knowledge that
contains a high degree of trust. The prime novelty of DeepTrust is the integration of a series of state-of-the-art NLP
techniques in retrieving information from a noisy Twitter data stream, and assessing information reliability from
various aspects, including the argumentation structure, evidence validity, neural generated text traces, and text
subjectivity.

The DeepTrust is comprised of three interconnected modules:

- Anomaly Detection module
- Information Retrieval module
- Reliability Assessment module

All modules function in sequential order within the DeepTrust framework, and jointly contribute to achieving an overall
high level of precision in retrieving information from Twitter that constitutes a collection of trusted knowledge to
explain financial anomalies. Solution effectiveness will be evaluated both module-wise and framework-wise to empirically
conclude the practicality of the DeepTrust framework in fulfilling its objective.

## Command-line Interface Usages

Retrieve a list of anomalies in `TWTR` (Twitter) pricing data between `04/01/2021` and `20/05/2021` using ARIMA-based
detection method.

```bash
python main.py -m AD -t TWTR -sd 04/01/2021 -ed 20/05/2021 --ad_method arima
```

Collect correlated tweets from Twitter data stream of `TSLA` (Tesla) regards to a detected financial anomaly on 22 Feb

2021. Data will be stored in the MongoDB database as specified in the `config.ini` file.

```bash
python main.py -m IR -ad 22/02/2021 -t TSLA
```

## Important Notes

Change following code in ```modeling_gpt.py``` in package ```pytorch-pretrained-bert``` to include GPT-2 Large
capabilities

```python
PRETRAINED_MODEL_ARCHIVE_MAP = {"gpt2":        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
                                "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
                                "gpt2-large":  "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
                                "gpt2-xl":     "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin"
                                }
PRETRAINED_CONFIG_ARCHIVE_MAP = {"gpt2":        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json",
                                 "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json",
                                 "gpt2-large":  "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-config.json",
                                 "gpt2-xl":     "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-config.json"
                                 }
```

Same for ```tokenization_gpt2.py``` in package ```pytorch-pretrained-bert``` to include GPT-2 Large capabilities

```python
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'gpt2':        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
    "gpt2-large":  "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
    "gpt2-xl":     "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json"
}
PRETRAINED_MERGES_ARCHIVE_MAP = {
    'gpt2':        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
    "gpt2-large":  "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
    "gpt2-xl":     "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt"
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'gpt2':        1024,
    'gpt2-medium': 1024,
    'gpt2-large':  1024,
    'gpt2-xl':     1024
}
```

## Future Plans

- [ ] Information Retrieval Modules
- [ ] Reliability Assessment Module

-----------

## Contributor

[Bill Chan -- Main Developer](https://github.com/billpwchan/)
