import torch
from tqdm import tqdm
from flair.data import Sentence

LABEL_IDX = {"group": 0, "product": 1, "corporation": 2, "location": 3, "person": 4, "creative-work": 5, "O": 6}
FILTER_POSTAGSET = ['NUM', 'DET', 'PRON', 'PUNCT', 'CCONJ', 'SYM', 'ADP', 'INTJ', 'PART']


def read_file(mode):
    print("Reading lines...")
    if mode == 'train':
        lines = open('./data/emerging.train.conll', encoding='utf-8').read().strip().split('\n')
    elif mode == 'dev':
        lines = open('./data/emerging.dev.conll', encoding='utf-8').read().strip().split('\n')
    elif mode == 'test':
        lines = open('./data/emerging.test.conll', encoding='utf-8').read().strip().split('\n')
    temp_sentence = []
    temp_tags = []
    sentence = []
    tag = []
    for l in lines:
        if l == '\t' or l == '':
            sentence.append(' '.join(temp_sentence))
            tag.append(' '.join(temp_tags))
            temp_sentence = []
            temp_tags = []
        else:
            l = l.split('\t')
            if len(l) == 2:
                temp_sentence.append(l[0])
                temp_tags.append(l[1])

    if len(temp_sentence) > 1:
        sentence.append(' '.join(temp_sentence))
        tag.append(' '.join(temp_tags))
    print("Reading has finished")
    return sentence, tag


def filter_data(postag_model, embedding, sentences, tags, vocab):
    tag_set = {}
    for sentence, tag in zip(sentences, tags):
        sentence_split = sentence.split()
        tag_split = tag.split()
        for i in range(len(sentence_split)):
            tag = tag_split[i]
            if tag != 'O':
                tag = tag[2:]
            if sentence_split[i] not in tag_set:
                tag_set[sentence_split[i]] = []
                tag_set[sentence_split[i]].append(tag)
            else:
                if tag not in tag_set[sentence_split[i]]:
                    tag_set[sentence_split[i]].append(tag)

    train_set = {}
    with torch.no_grad():
        for i in tqdm(range(len(sentences))):
            tokens = sentences[i].split()
            tag = tags[i].split()
            assert len(tokens) == len(tag)
            flair_sentence_posTag = Sentence(sentences[i])
            postag_model.predict(flair_sentence_posTag)
            entity = flair_sentence_posTag.get_spans('pos')
            flair_sentence_embed = Sentence(sentences[i])
            embedding.embed(flair_sentence_embed)
            for j in range(len(tag)):
                if len(tokens[j]) > 1 and tokens[j].isalpha() and len(tag_set[tokens[j]]) == 1 and tokens[j] in vocab:
                    if tag[j] != 'O':
                        tag[j] = tag[j][2:]
                    try:
                        assert str(entity[j].tokens[0]).split()[2] == tokens[j]
                        if entity[j].tag not in FILTER_POSTAGSET:
                            if tag[j] not in train_set:
                                train_set[tag[j]] = []
                                train_set[tag[j]].append((flair_sentence_embed[j].embedding.cpu(), LABEL_IDX[tag[j]],tag[j],tokens[j]))
                            else:
                                train_set[tag[j]].append((flair_sentence_embed[j].embedding.cpu(), LABEL_IDX[tag[j]],tag[j],tokens[j]))
                    except AssertionError:
                        continue
    return train_set


def resetSub(filename):
    lines = open('./result/emerging.test.conll', encoding='utf-8').read().strip().split('\n')
    temp_sentence = []
    temp_tags = []
    sentences = []
    tags = []
    for l in lines:
      if l == '\t' or l == '':
          sentences.append(temp_sentence)
          tags.append(temp_tags)
          temp_sentence = []
          temp_tags = []
      else:
          l = l.split()
          temp_sentence.append(l[0])
          temp_tags.append(l[1])
    sentences.append(temp_sentence)
    tags.append(temp_tags)
    print("Reading has finished")

    lines_predict = open(filename, encoding='utf-8').read().strip().split('\n')
    temp_tags = []
    tag_predict = []
    for l in lines_predict:
      if l == '\t' or l == '':
          if len(temp_sentence) > 0:
              tag_predict.append(temp_tags)
              temp_tags = []
      else:
          l = l.split(' ')
          if l[2][:2] == 'S-':
              l[2] = 'B-' + l[2][2:]
          elif l[2][:2] == 'E-':
              l[2] = 'I-' + l[2][2:]
          temp_tags.append(l[2])
    tag_predict.append(temp_tags)

    assert len(sentences) == len(tags) == len(tag_predict)

    w = open(filename,'w',encoding='utf-8')

    for sentence, predict_tag, truth_tag in zip(sentences,tags,tag_predict):
      for word,predict,truth in zip(sentence, predict_tag, truth_tag):
          w.write(word+'\t'+predict+'\t'+truth+'\n')
      w.write('\n')
    w.close()