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

## How to Install

Open Anaconda Prompt in you computer, and type the following command to create an environment.

```commandline
conda env create -f environment.yml
```

To export current environment, use the following command

```commandline
conda env export > environment.yml
```

## Command-line Interface Usages

### Anomaly Detection Module Examples

Retrieve a list of anomalies in `TWTR` (Twitter Inc.) pricing data between `04/01/2021` and `20/05/2021` using
ARIMA-based detection method.

```bash
python main.py -m AD -t TWTR -sd 04/01/2021 -ed 20/05/2021 --ad_method arima
```

The date format for both `-sd` and `-ed` parameters follows UK time format.

Available `--ad_method` includes `['arima', 'lof', 'if]`, which stands for `AUTO-ARIMA`, `Local Outlier Factor` and
`Isolation Forest`.

### Information Retrieval Module Examples

Collect correlated tweets from Twitter data stream of `TWTR` (Twitter Inc.) regards to a detected financial anomaly on
30 April 2021. Data will be stored in the MongoDB database as specified in the `config.ini` file.

```bash
python main.py -m IR -t TWTR -ad 30/04/2021
```

### Reliability Assessment Module Examples

1. **Feature-based Filtering**

Feature-based filtering on the retrieved collection of tweets (e.g., Remove tweets with no public metrics -
Retweets/Likes/Quotes). Rules can be specified in the `config.ini` under `RA.Feature.Config`.

```bash
python main.py -m RA -ad 30/04/2021 -t TWTR -rat feature-filter
```

2. **Synthetic Text Filtering**

*(Note: Synthetic Text Filtering only apply on tweets with Feature-Filter = True)*

Update `RoBERTa-based Detector`, `GLTR-BERT` and `GLTR-GPT-2` detectors results to MongoDB collection first.
```bash
python main.py -m RA -ad 30/04/2021 -t TWTR -rat neural-update -models gpt-2 gltr-bert gltr-gpt2
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

Update ```trainer.py``` for script ```run_clm.py``` for handling NaN loss values. 

```python
training_loss = self._training_step(model, inputs, optimizer)
tr_loss += 0 if np.isnan(training_loss) else training_loss
```

In ```_prediction_loop``` function

```python
temp_eval_loss = step_eval_loss.mean().item()
eval_losses += [0 if np.isnan(temp_eval_loss) else temp_eval_loss]
```

To fine-tune GPT-2-medium for Tweets

```commandline
python run_clm.py --model_name_or_path gpt2-medium --model_type gpt2 --train_data_file ./detector_dataset/TWTR_2021-04-30_train.txt --eval_data_file ./detector_dataset/TWTR_2021-04-30_test.txt --line_by_line --do_train --do_eval --output_dir ./tmp --overwrite_output_dir --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 5e-5 --save_steps 20000 --logging_steps 50 --num_train_epochs 1
```

## Future Plans

- [x] Anomaly Detection module
- [x] Information Retrieval Module
- [ ] Reliability Assessment Module

-----------

## Contributor

[Bill Chan -- Main Developer](https://github.com/billpwchan/)
