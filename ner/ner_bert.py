from flair.data import Corpus
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, StackedEmbeddings,TransformerWordEmbeddings
from typing import List
from torch.optim.adam import Adam

corpus: Corpus = WNUT_17()

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
  # TransformerWordEmbeddings('bert-base',layers='-1',fine_tune=True),  # bert-base
  # TransformerWordEmbeddings('bert-large',layers='-1',fine_tune=True),  # bert-large
  TransformerWordEmbeddings('vinai/bertweet-base',layers='-1',fine_tune=True),  # bertweet
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                     embeddings=embeddings,
                     tag_dictionary=tag_dictionary,
                     tag_type=tag_type,
                     use_crf=False,
                     use_rnn=False,
                     reproject_embeddings=False
                    )

# initialize trainer
from flair.trainers import ModelTrainer
# optimizer = Adam
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer = Adam)

trainer.train('../resources/taggers/example-ner',
       train_with_dev=False,
       learning_rate = 1e-5,
       min_learning_rate = 1e-5,
      )