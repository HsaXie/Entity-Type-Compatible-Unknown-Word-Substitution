import torch
from utils import read_file
from tqdm import tqdm
import flair
from flair.data import Sentence
from collections import Counter
import torch.nn.functional as F

flair.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def substitute(bertEncoder, all_tensor, all_word, all_tags,args):
    model = torch.load('./dml_model/final_model.pt')
    model = model.to(device)
    model.eval()

    dataset_sub = open('./data/testset_sub_info.txt', encoding='utf-8').read().strip().split('\n')

    tensor_allocate = []
    word_allocate = []
    buffer = []

    with torch.no_grad():
        for i in tqdm(range(len(all_tensor))):
            tensor_allocate.append(model.get_embedding(all_tensor[i].view(1,args.input_size).cuda()).cpu())
            word_allocate.append(all_word[i])
    tensor_dist = torch.zeros(len(tensor_allocate), args.output_size)
    for i, tensor in enumerate(tensor_allocate):
        tensor_dist[i] = tensor_allocate[i]
    tensor_dist = tensor_dist.cuda()

    for data in tqdm(dataset_sub):
        sentence = data.split('\t')[0]
        sentence_id = int(data.split('\t')[1])
        word_id = int(data.split('\t')[2])
        flair_sentence = Sentence(sentence)
        bertEncoder.embed(flair_sentence)
        try:
            embedding = model.get_embedding(flair_sentence[word_id].embedding.view(1, args.input_size))
        except IndexError:
            print(sentence + '\t' + str(word_id))
            continue
        tensor_dist_split = tensor_dist.cuda()
        embedding = embedding.cuda()
        dist = F.pairwise_distance(embedding, tensor_dist_split, 2)

        min_dist, idx = torch.topk(dist, largest=False, k=args.k)
        best_word = [word_allocate[index.item()] for index in idx]
        predict_tag = [all_tags[index.item()] for index in idx]
        predict_count = Counter(predict_tag)
        top = predict_count.most_common()
        predict = []
        max = top[0][1]
        for i in range(len(predict)):
            if top[i][1] == max:
                predict.append(top[i][0])
        predict = top[0][0]
        for index in idx:
            if all_tags[index] in predict:
                best_word = [word_allocate[index.item()]]
                break
        buffer.append((str(sentence_id), str(word_id), best_word[0]))

    sentence_test, tag_test = read_file('test')
    for replace in buffer:
        replace_id = int(replace[0])
        replace_word_idx = int(replace[1])
        replace_word = replace[2]
        origin_sentence_split = sentence_test[replace_id].split(' ')
        origin_sentence_split[replace_word_idx] = replace_word
        sentence_test[replace_id] = ' '.join(t for t in origin_sentence_split)

    f = open('./result/emerging.testSubstituted.conll', 'w',encoding='utf-8')

    for i in tqdm(range(len(sentence_test))):
        sentence = sentence_test[i].split()
        tag = tag_test[i].split()
        assert len(sentence) == len(tag)
        for j in range(len(sentence)):
                f.write(sentence[j] + '\t' + tag[j] + '\n')
        f.write('\n')

    f.close()