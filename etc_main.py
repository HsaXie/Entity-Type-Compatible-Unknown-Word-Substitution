from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from trainer import train
from substitute import substitute
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument("-em", "--encoder_mode", help="choosing encoder mode, we provide BERTWEET, BERT-BASE and BERT-LARGE", type=str, required=False, default='BERTWEET')
parser.add_argument("-e", "--epoch", help="Number of epochs", type=int, required=False, default=50)
parser.add_argument("-li", "--log_interval", help="Print intervals during training", type=str, required=False, default=1000)
parser.add_argument("-lr", "--lr", help="Learning rate", type=float, required=False, default=1e-4)
parser.add_argument("-drop", "--drop_out", help="Drop out", type=float, required=False, default=0.5)
parser.add_argument("-margin", "--margin", help="Triplet Loss Margin", type=float, required=False, default=1)
parser.add_argument("-bs", "--batch_size", help="Train batch size for model", type=int, required=False, default=16)
parser.add_argument("-k", "--k_num", help="Model will select the best suitable ETC word from k candidates. In paper, we use 1. And we encourage you to try different numbers to achieve better result.", type=int, required=False, default=1)
parser.add_argument("-osp", "--o_sample_prob", help="The probability of sampling O label", type=float, required=False, default=0.02)
parser.add_argument("-hs", "--hidden_size", help="ETC model hidden size", type=int, required=False, default=512)
parser.add_argument("-os", "--output_size", help="ETC model output size", type=int, required=False, default=256)

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.encoder_mode == 'BERT-LARGE':
    args.input_size = 1024
else:
    args.input_size = 768


if args.encoder_mode == 'BERTWEET':
    bertEncoder = TransformerWordEmbeddings('vinai/bertweet-base', layers='-1', pooling_operation='first')
elif args.encoder_mode =='BERT-BASE':
    bertEncoder = TransformerWordEmbeddings('bert-base-uncased', layers='-1', pooling_operation='first')
elif args.encoder_mode =='BERT-LARGE':
    bertEncoder = TransformerWordEmbeddings('bert-large-uncased', layers='-1', pooling_operation='first')


# the model can be downloaded from https://nlp.informatik.hu-berlin.de/resources/models/upos/en-pos-ontonotes-v0.4.pt
posTagModel = SequenceTagger.load('en-pos-ontonotes-v0.4.pt')


def main():
    all_tensor,  all_word, all_tags = train(bertEncoder, posTagModel, args)
    substitute(bertEncoder, all_tensor, all_word, all_tags, args)

if __name__ == '__main__':
    main()