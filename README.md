# ETC

Code for our  ICASSP 2021 paper : "[Improving NER in Social Media via Entity Type-Compatible Unknown Word Substitution](https://ieeexplore.ieee.org/document/9414304)"

#### OS:

Distributor ID:	Ubuntu
Description:	Ubuntu 16.04.1 LTS
Release:	16.04

#### GPU:

NVIDIA Tesla V100

#### Language:

Python 3.6.9

#### Required packages:

```
pip install flair==0.5.1
pip install transformers==3.2.0
pip install torch==1.6.0 torchvision==0.7.0
```

OR install with:

> pip install -r requirements.txt


### Evaluation Metrics:

>  Systems are evaluated using a modified version of conlleval.py, provided by WNUT-17 Committee

> [F1 Score](https://noisy-text.github.io/2017/files/wnuteval.py)



### Resources:

The datasets can be downloaded from (https://noisy-text.github.io/2017/emerging-rare-entities.html)

The crawl vocab is extracted from [fastText Crawl](https://fasttext.cc/docs/en/english-vectors.html)

The POS tagging model can be downloaded from (https://nlp.informatik.hu-berlin.de/resources/models/upos/en-pos-ontonotes-v0.4.pt)

### Run:

#### NER WordEmbedding baseline model 
> python ./ner/ner_wordEmbedding.py

#### NER BERT baseline model 
> python ./ner/ner_bert.py

#### ETC Substitution 
> python etc_main.py

#### Evaluation 
> python eval.py

#### fastText Crawl Model Type converting
> python ./embedding/vec2bin.py

