#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:49:54 2020

@author: vschmalz
"""


import torch
from torch.utils import data
import numpy as np
from model.model import IE_classifier
from dataset_fbank import fsc_data
import torch.nn
from torch.autograd import Variable
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type=str, help="model name", required=True)
parser.add_argument('-b', '--blocks', type=int, help='blocks', default=5)
parser.add_argument('-r', '--repeats', type=int, help='repeats', default=2)
parser.add_argument('-w', '--workers', type=int, help='workers', default=0)
parser.add_argument('-p', '--pathdataset', type=str, help='pathdataset')
parser.add_argument('--batch_size', type=int, help='pathdataset', default=2)
parser.add_argument('--mode', type=str, help='joint or multitask', default='joint', choices=['joint', 'multitask'])

# storing params
arg = parser.parse_args()
model_name = arg.model
path_dataset = arg.pathdataset
batch_size = arg.batch_size
mode = arg.mode

test_data = fsc_data(path_dataset + '/data/test_data.csv', max_len=64000)
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': arg.workers}
test_set_generator = data.DataLoader(test_data, **params)

valid_data = fsc_data(path_dataset + '/data/valid_data.csv', max_len=64000)
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': arg.workers}
valid_set_generator = data.DataLoader(valid_data, **params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)
if mode == 'multitask':
    model = IE_classifier(n_blocks=arg.blocks, n_repeats=arg.repeats, out_chan=(6, 14, 4))
else:
    model = IE_classifier(in_chan=512, n_blocks=arg.blocks, n_repeats=arg.repeats, out_chan=(31,))
# model = TCN(n_blocks=arg.blocks,n_repeats=arg.repeats,out_chan=n_classes)

# loading the model
model.load_state_dict(torch.load(model_name))
model.eval()
model.to(device)

intentSet, subIntentSet = test_data.getsets()
actionSet, objectSet, locationSet = subIntentSet


correct_test = []
for i, d in enumerate(test_set_generator):
    feat, label = d
    print('Iter %d (%d/%d)' % (i, i * batch_size, len(test_data)), end='\r')
    #z_eval = model(feat.float().squeeze(dim=1).to(device))
    z_eval = model(feat.float().to(device))
    #print(feat.shape)
    #print(z_eval[0].shape)
    #print(z_eval)
    exit
    pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
    if mode == 'multitask':
        combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in
                         range(len(pred[0]))]
        pred_test = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
    else:
        pred_test = pred[0]

    correct_test.append(np.array(pred_test == label[0], dtype=float))

acc_test = (np.mean(np.hstack(correct_test)))
print("The accuracy on test set is %f" % acc_test)

correct_valid = []
for i, d in enumerate(valid_set_generator):
    print('Iter %d (%d/%d)' % (i, i * batch_size, len(valid_data)), end='\r')
    feat, label = d
    a_eval = model(feat.float().to(device)) #squeeze(dim = 1).
    # a_eval = model(feat.float().to(device))
    pred = [torch.max(a.detach().cpu(), dim=1)[1] for a in a_eval]
    if mode == 'multitask':
        combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in
                         range(len(pred[0]))]
        pred_test = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
    else:
        pred_test = pred[0]

    correct_valid.append(np.array(pred_test == label[0], dtype=float))
acc_val = (np.mean(np.hstack(correct_valid)))
print("The accuracy on the validation set is %f" % acc_val)
'''
# reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type=str, help="model name", required=True)
parser.add_argument('-b', '--blocks', type=int, help='blocks', default=5)
parser.add_argument('-r', '--repeats', type=int, help='repeats', default=2)
parser.add_argument('-w', '--workers', type=int, help='workers', default=2)
parser.add_argument('-p', '--pathdataset', type=str, help='pathdataset')
parser.add_argument('--batch_size', type=int, help='pathdataset', default=100)
parser.add_argument('--mode', type=str, help='joint or multitask',default='multitask',choices=['joint', 'multitask'])

# storing params
arg = parser.parse_args()
model_name = arg.model
path_dataset = arg.pathdataset
batch_size = arg.batch_size
mode = arg.mode

test_data = fsc_data(path_dataset + '/data/test_data.csv', max_len=64000, signaltype='wavs')
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': arg.workers}
test_set_generator = data.DataLoader(test_data, **params)

valid_data = fsc_data(path_dataset + '/data/valid_data.csv', max_len=64000, signaltype='wavs')
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': arg.workers}
valid_set_generator = data.DataLoader(valid_data, **params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)
if mode == 'multitask':
	model = TCN(n_blocks=arg.blocks, n_repeats=arg.repeats, out_chan=(6, 14, 4))
else:
	model = TCN(n_blocks=arg.blocks, n_repeats=arg.repeats, out_chan=(31,))
# model = TCN(n_blocks=arg.blocks,n_repeats=arg.repeats,out_chan=n_classes)

# loading the model
model.load_state_dict(torch.load(model_name))
model.eval()
model.to(device)

intentSet, subIntentSet = test_data.getsets()
actionSet, objectSet, locationSet = subIntentSet


###Dovrei avere un solo codice per non fare cazzate.....
correct_test = []
for i, d in enumerate(test_set_generator):
	feat, label = d
	print('Iter %d (%d/%d)' % (i, i * batch_size, len(test_data)), end='\r')

	z_eval = model(feat.float().to(device))
	pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
	if mode == 'multitask':
		combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in range(len(pred[0]))]
		pred_test = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
	else:
		pred_test = pred[0]

	correct_test.append(np.array(pred_test == label[0], dtype=float))

acc_test = (np.mean(np.hstack(correct_test)))
print("The accuracy on test set is %f" % (acc_test))

correct_valid = []
for i, d in enumerate(valid_set_generator):
	print('Iter %d (%d/%d)' % (i, i * batch_size, len(valid_data)), end='\r')
	feat, label = d

	a_eval = model(feat.float().to(device))
	pred = [torch.max(a.detach().cpu(), dim=1)[1] for a in a_eval]
	if mode == 'multitask':
		combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in range(len(pred[0]))]
		pred_test = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
	else:
		pred_test = pred[0]

	correct_valid.append(np.array(pred_test == label[0], dtype=float))
acc_val = (np.mean(np.hstack(correct_valid)))
print("The accuracy on the validation set is %f" % (acc_val))
'''
