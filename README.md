# Analysis of Vocabulary and Subword Tokenization Settings for Optimal Fine-tuning of MT

**Accepted at RANLP 2025**

This repository contains the code and data for our RANLP 2025 paper, which systematically investigates how different vocabulary and subword (BPE) tokenization strategies impact the fine-tuning of neural machine translation (NMT) models for domain adaptation. Using English-German general and medical domain datasets, we show that optimal fine-tuning is achieved by deriving both vocabulary and BPE from in-domain data, while ensuring good coverage of the original model’s vocabulary. The results provide practical guidelines for adapting NMT systems to new domains.

## Authors

**Javad Pourmostafa Roshan Sharami**, **Dimitar Shterionov**, **Pieter Spronck**  
Department of Cognitive Science & Artificial Intelligence, Tilburg University, The Netherlands  

Emails:  
- [j.pourmostafa@tilburguniversity.edu](mailto:j.pourmostafa@tilburguniversity.edu)  
- [d.shterionov@tilburguniversity.edu](mailto:d.shterionov@tilburguniversity.edu)  
- [p.spronck@tilburguniversity.edu](mailto:p.spronck@tilburguniversity.edu)
