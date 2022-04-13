import os
from torch.utils.data import Dataset
import librosa
import numpy as np


class WaveformDataset(Dataset):
    def __init__(self, dataset, csv_filename, limit=None, offset=0, sample_length=16384):
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.intent = []
        self.audio_id = []
        self.transcription = []
        self.subintent = [[] for i in range(3)]
        with open(os.path.abspath(os.path.expanduser(csv_filename)), "r") as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audio_id.append(items[1])
                self.intent.append(tuple(items[-3:]))
                for i in range(3):
                    self.subintent[i].append(self.intent[-1][i])
        intent_set = sorted(list(set(self.intent)))
        self.intent_labels = [intent_set.index(t) for t in self.intent]
        subintent_sets = [sorted(list(set(self.subintent[i]))) for i in range(3)]
        self.subintent_labels = []
        for i in range(3):
            self.subintent_labels.append([subintent_sets[i].index(t) for t in self.subintent[i]])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)
        if len(mixture) < 64000:
            mixture = np.pad(mixture, [(0, 64000 - mixture.shape[0]), ], mode='constant')
        label = (self.intent_labels[item], [self.subintent_labels[i][item] for i in range(3)])
        return mixture.reshape(1, -1), name, label
        
    def getsets(self):
        return sorted(list(set(self.intent))), [sorted(list(set(self.subintent_labels[i]))) for i in range(3)]
