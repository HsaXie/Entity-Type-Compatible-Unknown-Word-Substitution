from flair.data import Corpus
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus: Corpus = WNUT_17()

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# glove
# embedding_types: List[TokenEmbeddings] = [
#     WordEmbeddings('glove'),
# ]

# glove
# the crawl embedding is provided by https://fasttext.cc/docs/en/english-vectors.html
# embedding_types: List[TokenEmbeddings] = [
#   WordEmbeddings('./embedding/crawl-300d-2M.bin'),
# ]

# stacked
# the stacked embeddings are provided by https://github.com/flairNLP/flair and the crawl vocabulary was compressed to the commonly used 1M words by flair.
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('crawl'),
    WordEmbeddings('twitter'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              train_with_dev=False)