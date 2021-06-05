<div align="center">
  <img alt="DeepTrust Logo" src="https://raw.githubusercontent.com/billpwchan/DeepTrust/master/docs/img/logo.png" width="400px" />

**billpwchan/DeepTrust API Reference Documentation**
</div>

## DeepTrust Description

Different from existing works, the present project proposes a reliable information extraction framework named DeepTrust. DeepTrust enables financial data providers to precisely locate correlated information on Twitter upon a financial anomaly occurred, and apply information retrieval and validation techniques to preserve only reliable knowledge that contains a high degree of trust. The prime novelty of DeepTrust is the integration of a series of state-of-the-art NLP techniques in retrieving information from a noisy Twitter data stream, and assessing information reliability from various aspects, including the argumentation structure, evidence validity, neural generated text traces, and text subjectivity. 

The DeepTrust is comprised of three interconnected modules: 
- Anomaly Detection module
- Information Retrieval module
- Reliability Assessment module 
  
All modules function in sequential order within the DeepTrust framework, and jointly contribute to achieving an overall high level of precision in retrieving information from Twitter that constitutes a collection of trusted knowledge to explain financial anomalies. Solution effectiveness will be evaluated both module-wise and framework-wise to empirically conclude the practicality of the DeepTrust framework in fulfilling its objective. 

## Command-line Interface Usages

Retrieve a list of anomalies in `TWTR` (Twitter) pricing data between `04/01/2021` and `20/05/2021` using ARIMA-based 
detection method.
```bash
python main.py -m AD -t TWTR -sd 04/01/2021 -ed 20/05/2021 --ad_method arima
```

## Future Plans

- [ ] Information Retrieval Modules
- [ ] Reliability Assessment Module

-----------

## Contributor

[Bill Chan -- Main Developer](https://github.com/billpwchan/)
