from triplet import TripletLoss,TripletNet,TripletWnut
import flair
import torch.optim as optim
from utils import *
import random
from torch.utils.data import DataLoader
import numpy as np

flair.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = open('./data/crawl_vocab.txt', encoding='utf-8').read().strip().split('\n')


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        optimizer.zero_grad()
        outputs = model(data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def train(bert_encoder,posTag_model,args):
    sentences_train, tags_train = read_file('train')
    train_dict = filter_data(posTag_model,bert_encoder, sentences_train, tags_train,vocab)
    model = TripletNet(args.input_size, args.hidden_size, args.output_size, args.drop_out)
    loss_fn = TripletLoss(args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        train_set = list()
        for key in train_dict:
            if key != 'O':
                train_set += train_dict[key]
            else:
                train_set += random.sample(train_dict[key], int(len(train_dict[key]) * args.o_sample_prob))
        triplet_train_dataset = TripletWnut(train_set)
        train_loader = DataLoader(triplet_train_dataset, args.batch_size, shuffle=True)
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, args.log_interval, metrics=[])

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, args.epoch, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)
    torch.save(model, './dml_model/final_model.pt')
    train_set = list()
    for key in train_dict:
        train_set += train_dict[key]
    all_tensor = []
    all_word = []
    all_tags = []
    for data in train_set:
        all_tensor.append(data[0])
        all_tags.append(data[2])
        all_word.append(data[3])
    assert len(all_tensor) == len(all_tags) == len(all_word)

    return all_tensor,  all_word, all_tags
