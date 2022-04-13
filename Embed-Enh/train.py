import torch
from torch.utils import data
from model import TCN, SEModel
import numpy as np
from unet import UNet
from dataset import Dataset
import torch.optim as optim
import torch.nn
import argparse
import os
import copy
from torch.autograd import Variable


os.environ['TF_CPP_MIN_LOF_LEVEL'] = '3'
parser = argparse.ArgumentParser(description='Description')
parser.add_argument('-m', '--back_model', type=str, help='back_model name', default='best_bkmodel2.pkl')
parser.add_argument('-M', '--front_model', type=str, help='front_model name', default='best_frmodel2.pkl')
parser.add_argument('-b', '--blocks', type=int, help='blocks', default=5)
parser.add_argument('-r', '--repeats', type=int, help='repeats', default=2)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning_rate', default=0.001)
parser.add_argument('-e', '--epochs', type=int, help='epochs', default=400)
parser.add_argument('-w', '--workers', type=int, help='workers', default=0)
parser.add_argument('--batch_size', type=int, help='batch_size', default=10)
parser.add_argument('--mode', type=str, help='joint or multitask', default='joint', choices=['joint', 'multitask'])

torch.set_num_threads(1)

arg = parser.parse_args()
numworkers = arg.workers
tcnBlocks = arg.blocks
tcnRepeats = arg.repeats
learning_rate = arg.learning_rate
epochs = arg.epochs
frmodelname = arg.front_model
bkmodelname = arg.back_model
batch_size = arg.batch_size
mode = arg.mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

train_data = Dataset(dataset='/data/mohamed/second_strat_mod/fulltr.txt',
                     csv_filename="/data/mohamed/data_emb/clean_fluent_speech_commands_dataset/datapt/train_data.csv",
                     sample_length=450)
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': numworkers}
train_set_generator = data.DataLoader(train_data, **params)

valid_data = Dataset(dataset='/data/mohamed/second_strat_mod/fullvl.txt',
                     csv_filename="/data/mohamed/data_emb/clean_fluent_speech_commands_dataset/datapt/valid_data.csv",
                     sample_length=450)
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': numworkers}
valid_set_generator = data.DataLoader(valid_data, **params)

front_model = SEModel().to(device)

if mode == 'multitask':
    back_model = TCN(in_chan=512, n_blocks=tcnBlocks, n_repeats=tcnRepeats, out_chan=(6,14,4)).to(device)
else:
    back_model = TCN(in_chan=512, n_blocks=tcnBlocks, n_repeats=tcnRepeats, out_chan=(31,)).to(device)


optimizer_front = optim.Adam(front_model.parameters(), lr=0.0001)
optimizer_back = optim.Adam(back_model.parameters(), lr=learning_rate)

criterion_front = torch.nn.MSELoss()  
criterion_back = torch.nn.CrossEntropyLoss()

best_accuracy = 0
loss_total = 0.0
front_total = 0.0
back_total = 0.0
intentSet, subIntentSet = train_data.getsets()
actionSet, objectSet, locationSet = subIntentSet

for e in range(epochs):
    for i, (mixture, clean, name, label) in enumerate(train_set_generator):
        front_model.train()
        back_model.train()
        mixture = mixture.to(device, dtype=torch.float)
        clean = clean.to(device, dtype=torch.float)
        enhanced = front_model(mixture.float())

        enhanced = enhanced.to(device)

        front_loss = criterion_front(clean, enhanced)  

        y = back_model(enhanced.float().to(device))
        if mode == 'multitask':
            back_loss = torch.stack([criterion_back(y[i], label[1][i].to(device)) for i in range(len(y))]).sum()
        else:
            back_loss = criterion_back(y[0], label[0].to(device))

        alpha = 0.5
        loss = ((1 - alpha) * back_loss) + (alpha * front_loss)

        print("Iteration %d in epoch %d --> front_loss = %f  back_loss = %f loss_total = %f" % (i, e, front_loss.item(),
                                                                                                back_loss.item(),
                                                                                                loss.item()
                                                                                                ), end='\r')
        loss_total += loss.item()
        front_total += front_loss.item()
        back_total += back_loss.item()
        loss.backward()

        optimizer_front.step()
        optimizer_back.step()

        optimizer_front.zero_grad()
        optimizer_back.zero_grad()

        if i % 100 == 0:
            front_model.eval()
            back_model.eval()
            if mode == 'multitask':
                correct = [[] for i in range (4)]
            else:
                correct = []

            for j, (mixture, clean, name, label) in enumerate(valid_set_generator):
                mixture = mixture.to(device)
                clean = clean.to(device)

                enhanced = front_model(mixture.float())
                enhanced = enhanced.to(device)


                y_pred = back_model(enhanced.float().to(device))
                if mode == 'multitask':
                    pred = [torch.argmax(y.detach().cpu(), dim=1) for y in y_pred]
                    combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in range(len(pred[0]))]
                    intent_pred = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
                    for p in range(len(correct) - 1):
                        correct[p].append((pred[p] == label[1][p]).float())
                    correct[-1].append((intent_pred == label[0]).float())
                else:
                    pred = torch.argmax(y_pred[0].detach().cpu(), dim=1)
                    intent_pred = pred
                    correct.append((intent_pred == label[0]).float())

                if j > 100:
                    break
            if mode == 'multitask':
                acc = [np.mean(np.hstack(c)) for c in correct]
                intent_acc = acc[-1]
            else:
                acc = np.mean(np.hstack(correct))
                intent_acc = acc
            iter_acc = '\n iteration %d epoch %d -->' % (i, e)
            print(iter_acc, acc, best_accuracy)

            if intent_acc > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (intent_acc, best_accuracy)
                print(improved_accuracy)
                best_accuracy = intent_acc
                best_epoch = e
                best_bkmodel = copy.deepcopy(back_model)
                torch.save(best_bkmodel.state_dict(), bkmodelname)


    best_frmodel = copy.deepcopy(front_model)
    torch.save(best_frmodel.state_dict(), frmodelname)