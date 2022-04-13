import os
import librosa
from torch.utils import data
import numpy as np
from util.utils import sample_fixed_length_data_aligned
import soundfile as sf
import torch


class Dataset(data.Dataset):
    def __init__(self, dataset, csv_filename, limit=None, offset=0, sample_length=16384, max_len=64000, mode="train"):
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        assert mode in ("train", "validation")
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode
        self.intent = []
        self.audio_id = []
        self.transcription = []
        self.max_len = max_len
        self.subintent = [[] for i in range(3)]

        with open(os.path.abspath(os.path.expanduser(csv_filename)), "r") as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audio_id.append(items[1])
                self.intent.append(tuple(items[-3:]))
                for i in range(3):
                    self.subintent[i].append(self.intent[-1][i])
        #utterance_set = sorted(list(set(self.transcriptions)))            
        intent_set = sorted(list(set(self.intent)))
        self.intent_labels = [intent_set.index(t) for t in self.intent]
        subintent_sets = [sorted(list(set(self.subintent[i]))) for i in range(3)]
        self.subintent_labels = []
        for i in range(3):
            self.subintent_labels.append([subintent_sets[i].index(t) for t in self.subintent[i]])


    def __len__(self):
        return len(self.audio_id)   

    def __getitem__(self,item,sample_length=16384):
        audio_file = self.audio_id[item]
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)
        clean, sr = sf.read('/data/mohamed/fluent_speech_commands_dataset/' + audio_file)
        clean = clean[:self.max_len]
        if len(clean) < self.max_len:
            clean = np.pad(clean, [(0, self.max_len - clean.shape[0]), ], mode='constant')
        if len(mixture) < self.max_len:
            mixture = np.pad(mixture, [(0, self.max_len - mixture.shape[0]), ], mode='constant')
        label = (self.intent_labels[item], [self.subintent_labels[i][item] for i in range(3)])
        if len(mixture) > self.max_len or len(clean) > self.max_len:
            clean = clean[:self.max_len]
            mixture = mixture[:self.max_len]
        assert len(mixture) == len(clean)             
        return mixture.reshape(1, -1), clean.reshape(1, -1) , filename, label  
        
    def getsets(self):
        return sorted(list(set(self.intent))), [sorted(list(set(self.subintent_labels[i]))) for i in range(3)]                           

