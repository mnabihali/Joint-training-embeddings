import os
import librosa
from torch.utils import data
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F



class Dataset(data.Dataset):
    def __init__(self, dataset, csv_filename, limit=None, offset=0, sample_length=450):
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.transcription = []
        self.subintent = [[] for i in range(3)]
        self.eps = np.finfo(np.float64).eps
        self.intent = []
        self.audio_id = []
        with open(csv_filename, encoding='utf') as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audio_id.append(items[1])
                if (len(items)) == 7:
                    self.transcription.append(items[3])
                else:
                    self.transcription.append((" ").join(items[3:5]))
                self.intent.append(tuple(items[-3:]))
                for i in range(3):
                    self.subintent[i].append(self.intent[-1][i])
            utteranceset = sorted(list(set(self.transcription)))
            self.sentence_labels = [utteranceset.index(t) for t in self.transcription]
            intent_set = sorted(list(set(self.intent)))
            self.intent_labels = [intent_set.index(t) for t in self.intent]
            subintent_sets = [sorted(list(set(self.subintent[i]))) for i in range(3)]
            self.subintent_labels = []
            for i in range(3):
                self.subintent_labels.append([subintent_sets[i].index(t) for t in self.subintent[i]])

    def __len__(self):
        return len(self.audio_id)

    def __getitem__(self, item, sample_length=400):
        audio_file = self.audio_id[item]
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture = torch.load(os.path.abspath(os.path.expanduser(mixture_path)))
        clean = torch.load(os.path.abspath(os.path.expanduser(clean_path)))

        p1d = (0, 350 - clean.shape[-1])
        clean = F.pad(clean, p1d, "constant", 0)
        p2d = (0, 350 - mixture.shape[-1])
        mixture = F.pad(mixture, p2d, "constant", 0)
        mixture = mixture.squeeze(0)
        clean = clean.squeeze(0)
        assert len(clean) == len(mixture)
        label = (self.intent_labels[item], [self.subintent_labels[i][item] for i in range(3)])
        return mixture, clean, filename, label

    def getsets(self):
        return sorted(list(set(self.intent))), [sorted(list(set(self.subintent[i]))) for i in range (3)]


if __name__ == "__main__":
    mydata = Dataset(dataset="/data/disk1/data/mnabih/mnabih/second_strat/fulltr.txt",
                     csv_filename="/data/disk1/data/mnabih/mnabih/clean_fluent_speech_commands_dataset/datapt/train_data.csv")
    print(mydata)
    params = {'batch_size': 5, 'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)
    for x, y, z, v in test_set_generator:
        print(x.shape)
        print(y.shape)
        print(z)
        print(v)
        break

'''
import os
import librosa
from torch.utils import data
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F



class Dataset(data.Dataset):
    def __init__(self, dataset, csv_filename, limit=None, offset=0, sample_length=450):
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.intent = []
        self.audio_id = []
        with open(os.path.abspath(os.path.expanduser(csv_filename)), "r") as fcsv:
            lines = fcsv.readlines()
            for l in lines[1:]:
                items = l[:-1].split(',')
                self.audio_id.append(items[1])
                self.intent.append(tuple(items[-3:]))
        intent_set = sorted(list(set(self.intent)))
        self.intent_labels = [intent_set.index(t) for t in self.intent]

    def __len__(self):
        return len(self.audio_id)

    def __getitem__(self, item, sample_length=400):
        audio_file = self.audio_id[item]
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture = torch.load(os.path.abspath(os.path.expanduser(mixture_path)))
        clean = torch.load(os.path.abspath(os.path.expanduser(clean_path)))
        #f len(clean) != len(mixture):
        #if len(clean) < self.sample_length or len(mixture) < self.sample_length:
        p1d = (0, 350 - clean.shape[-1])
        clean = F.pad(clean, p1d, "constant", 0)
        p2d = (0, 350 - mixture.shape[-1])
        mixture = F.pad(mixture, p2d, "constant", 0)
        mixture = mixture.squeeze(0)
        clean = clean.squeeze(0)
        assert len(clean) == len(mixture)
        label = (self.intent_labels[item])
        return mixture, clean, filename, label

    def getsets(self):
        return sorted(list(set(self.intent)))


if __name__ == "__main__":
    mydata = Dataset(dataset="/home/mnabih/test/test.txt",
                     csv_filename="/home/mnabih/fluent_speech_commands_dataset/data/train_data.csv")
    print(mydata)
    params = {'batch_size': 5, 'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)
    for x, y, z, v in test_set_generator:
        print(x.shape)
        print(y.shape)
        print(z)
        print(v)
        break
'''