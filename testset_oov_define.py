from utils import *

sentence, tag = read_file('test')

vocab = open('./data/crawl_vocab.txt', encoding='utf-8').read().strip().split('\n')

w = open('./data/testset_sub_info.txt','w',encoding='utf-8')

idx = -1
cnt = 0

for s,t in zip(sentence,tag):
  idx+=1
  sentence_split = s.split()
  tag_split = t.split()
  for j in range(len(sentence_split)):
    if len(sentence_split[j])>1 and sentence_split[j].isalpha() and sentence_split[j] not in vocab:
      cnt+=1
      if tag_split[j] !='O':
        tag_split[j] = tag_split[j][2:]
      w.write(sentence[idx]+'\t'+str(idx)+'\t'+str(j)+'\t'+tag_split[j]+'\n')

w.close()

