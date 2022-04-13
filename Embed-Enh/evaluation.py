import torch
from torch.utils import data
from model import TCN , SEModel
import numpy as np
from dataset import Dataset
import torch.nn
import argparse
import os
from unet import UNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Destination')

parser.add_argument('-m', '--back_model', type=str, help='back_modelname', required=True)
parser.add_argument('-M', '--front_model', type=str, help='front_modelname', required=True)
parser.add_argument('-b', '--blocks', type=int, help='blocks', default=5)
parser.add_argument('-r', '--repeats', type=int, help='repeats', default=2)
parser.add_argument('-w', '--workers', type=int, help='worker', default=0)
parser.add_argument('--batch_size', type=int, help='batch_size', default=1)
parser.add_argument('--mode', type=str, help='joint or multitask',default='joint',choices=['joint', 'multitask'])

arg = parser.parse_args()
frmodelname = arg.front_model
bkmodelname = arg.back_model
batch_size = arg.batch_size
numworkers = arg.workers
tcnBlocks = arg.blocks
tcnRepeats = arg.repeats
mode = arg.mode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

test_data = Dataset(dataset='/data/mohamed/second_strat_mod/fullts.txt',
                    csv_filename="/data/mohamed/data_emb/clean_fluent_speech_commands_dataset/datapt/test_data.csv",
                    sample_length=450)
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': numworkers}
test_set_generator = data.DataLoader(test_data, **params)

front_model = SEModel() #UNet() #
back_model = TCN(in_chan=512, n_blocks= tcnBlocks, n_repeats= tcnRepeats, out_chan=(31,))

# Loading the model state_dict
front_model.load_state_dict(torch.load(frmodelname))
front_model.eval()
front_model.to(device)

back_model.load_state_dict(torch.load(bkmodelname))
back_model.eval()
back_model.to(device)

intentSet, subIntentSet = test_data.getsets()
actionSet, objectSet, locationSet = subIntentSet

correct_test = []
for i, (mixture, clean, name, label) in enumerate(test_set_generator):
    print('Iter %d (%d/%d)' % (i, i*batch_size, len(test_data)), end='\r')
    mixture = mixture.to(device)
    #mixture = torch.squeeze(mixture, dim=0)
    #print('mix shape', mixture.shape)
    enhanced = front_model(mixture.float())
    enhanced = enhanced.to(device)
    #print('en shape', enhanced.shape)
    z_eval = back_model(enhanced.float().to(device))
    pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
    if mode == 'multitask':
       combined_pred = [(actionSet[pred[0][l]], objectSet[pred[1][l]], locationSet[pred[2][l]]) for l in range(len(pred[0]))]
       pred_test = torch.tensor([intentSet.index(p) if p in intentSet else -1 for p in combined_pred])
    else:
       pred_test = pred[0]
    #if pred_test == label[0]:
       #print('yes')
    #else:
       #print('N0000000000000000000000000o')
    correct_test.append(np.array(pred_test == label[0], dtype=float))
    #y_pred = back_model(enhanced.float().to(device))

    #pred = torch.argmax(y_pred[0].detach().cpu(), dim=1)
    #intent_pred = pred

    #correct_test.append((intent_pred == label[0]).float())

acc_test = np.mean(np.hstack(correct_test))
print('The accracy on the test dataset is %f' % acc_test)


valid_data = Dataset(dataset='/data/mohamed/second_strat_mod/fullvl.txt',
                    csv_filename="/data/mohamed/data_emb/clean_fluent_speech_commands_dataset/datapt/valid_data.csv",
                    sample_length=450)
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': numworkers}
valid_set_generator = data.DataLoader(valid_data, **params)


correct_valid = []
for i, (mixture, clean, name, label) in enumerate(valid_set_generator):
    print('Iter %d (%d/%d)' % (i, i*batch_size, len(valid_data)), end='\r')
    mixture = mixture.to(device)
    #mixture = torch.squeeze(mixture, dim=0)
    #print('mix shape', mixture.shape)
    enhanced = front_model(mixture.float())
    enhanced = enhanced.to(device)
    #print('en shape', enhanced.shape)
    y_pred = back_model(enhanced.float().to(device))

    pred = torch.argmax(y_pred[0].detach().cpu(), dim=1)
    intent_pred = pred

    correct_valid.append((intent_pred == label[0]).float())

acc_valid = np.mean(np.hstack(correct_valid))
print('The accracy on the valid dataset is %f' % acc_valid)

