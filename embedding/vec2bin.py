import gensim
# this is designed for crawl
# the crawl-300d-2M.vec can be download in https://fasttext.cc/docs/en/english-vectors.html
word_embeddings = gensim.models.KeyedVectors.load_word2vec_format('./embedding/crawl-300d-2M.vec', binary=False)
word_embeddings.wv.save_word2vec_format('./embedding/crawl-300d-2M.bin',binary=True)