import argparse
import json
import os
#import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.utils import initialize_config, load_checkpoint, emb
import numpy as np
from model.model import IE_classifier
import gc
import soundfile as sf
from dataset.waveform_dataset_enhancement import WaveformDataset
import pandas as pd
gc.collect()



torch.cuda.empty_cache()
gc.collect()
parser = argparse.ArgumentParser(description="Joint_training")
parser.add_argument("-C", "--config", type=str, required=True, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", type=str, required=True, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="Checkpoint.")
parser.add_argument('-m', '--back_model', type=str, help="model name", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = json.load(open(args.config))
model_checkpoint_path = args.model_checkpoint_path
output_dir = args.output_dir
model_name = args.back_model
assert os.path.exists(output_dir), "----> Enhance Directory Should Exist"
torch.cuda.empty_cache()
gc.collect()

"""
Dataloader
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

test_data = WaveformDataset(csv_filename = "/data/mohamed/fluent_speech_commands_dataset/data/test_data.csv", dataset = '/data/mohamed/jt/code/ts.txt')
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)
intentSet, subIntentSet = test_data.getsets()
actionSet, objectSet, locationSet = subIntentSet


"""
Model
"""
# load front_end model
model = initialize_config(config["model"])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

# load back_end model
model_back = IE_classifier(in_chan=512, n_blocks=5, n_repeats=2, out_chan=(31,)).float()
model_back.load_state_dict(torch.load(model_name))
model_back.to(device)
model_back.eval()

model_back.to(device)

"""
Enhancement Stage
"""
stat = []
lab = []
correct_test = []
sample_length = config["custom"]["sample_length"]
for mixture, name, label in tqdm (dataloader):
    torch.cuda.empty_cache()
    gc.collect()
    assert len(name) == 1, "Only support batch_size = 1"
    name = name[0]
    padded_length = 0
    mixture = mixture.to(device)
    mixture = mixture.float()
    torch.cuda.empty_cache()
    gc.collect()
    if mixture.size(-1) % sample_length !=0:
        #torch.cuda.empty_cache()
        #gc.collect()
        padded_length = sample_length - (mixture.size(-1) % sample_length)
        mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)
    assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
    mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))
    #torch.cuda.empty_cache()
    #gc.collect()

    enhanced_chunks = []
    for chunk in mixture_chunks:
        #torch.cuda.empty_cache()
        #gc.collect()
        enhanced_chunks.append(model(chunk).float().detach().cpu())
        #torch.cuda.empty_cache()
        #gc.collect()
    enhanced = torch.cat(enhanced_chunks, dim=-1)
    #torch.cuda.empty_cache()
    #gc.collect()
    if padded_length !=0:
        #torch.cuda.empty_cache()
        #gc.collect()
        enhanced = enhanced[:,:,:-padded_length]
        mixture = mixture[:,:,:-padded_length]
        #torch.cuda.empty_cache()
        #gc.collect()
    else:
        enhanced = enhanced
        #torch.cuda.empty_cache()
        #gc.collect()
    enhanced = enhanced.to(device)
    enh = emb(enhanced).double()
    #p1d = (0, 350-enh.shape[-1])
        #print('The padding', padding_length)
    #enha = torch.nn.functional.pad(enh, p1d, mode='constant', value=0)
    #print(enha.shape)
    z_eval = model_back(enh.float().to(device))
    #torch.cuda.empty_cache()
    #gc.collect()
    pred = [torch.max(z.detach().cpu(), dim=1)[1] for z in z_eval]
    #torch.cuda.empty_cache()
    #gc.collect()
    pred_test = pred[0]
    ##print('The pred  is',pred_test)
    #act_label = label[0]
    ##stat.append(pred_test)
    #np.savetxt('output.csv', pred_test,delimiter= ',')
    #prediction = pd.DataFrame(stat, columns=['predictions']).to_csv('predictions05.csv')
    correct_test.append(np.array(pred_test == label[0], dtype=float))
    #torch.cuda.empty_cache()
    #gc.collect()

    #print('Iter %d (%d/%d)' % (i, i * batch_size, len(dataloader)), end='\r')

    enhanced = enhanced.reshape(-1).detach().cpu().numpy()
    output_path = os.path.join(output_dir, f"{name}.wav")
    sr = 16000
    sf.write(output_path, enhanced, sr)
    #librosa.output.write_wav(output_path, enhanced, sr=16000)
    #torch.cuda.empty_cache()
    #gc.collect()



acc_test = (np.mean(np.hstack(correct_test)))
print("The accuracy on test set is %f" % (acc_test))

torch.cuda.empty_cache()
gc.collect()
