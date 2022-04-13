import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from trainer.base_trainer import BaseTrainer
from util.utils import compute_PESQ, compute_STOI, emb
from model.model import IE_classifier
from dataset.waveform_dataset import Dataset
import torch
from torch.utils import data
import soundfile as sf
import torch.optim as optim
import torch.nn
from torch.autograd import Variable
import os
import torch.nn.functional as F
plt.switch_backend('agg')

device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device2)
torch.set_num_threads(1)
model_back = IE_classifier(in_chan=512, n_blocks=5, n_repeats= 2, out_chan=(31,)).to(device2)
optimizer_back = optim.Adam(model_back.parameters(), lr=0.001)

train_data = Dataset(csv_filename = "/data/mohamed/fluent_speech_commands_dataset/data/train_data.csv", dataset = '/data/mohamed/jt/code/fulltr.txt')
valid_data = Dataset(csv_filename = "/data/mohamed/fluent_speech_commands_dataset/data/valid_data.csv", dataset = '/data/mohamed/jt/code/fullvl.txt')

intentSet, subIntentSet = train_data.getsets()
actionSet, objectSet, locationSet = subIntentSet



best_acc = 0.0

best_accuracy = 0.0

class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, loss_function2 ,optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, loss_function2 ,optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.model = self.model.float()
        global best_acc
        global best_accuracy


    def _train_epoch(self, epoch):
        loss_total = 0.0
        front_total = 0.0
        back_total = 0.0
        corr = []
        correct = []
        global best_acc
        global best_accuracy
        for i, (mixture, clean, name, label) in enumerate(self.train_data_loader):
            padded_length = 0
            mixture = mixture.to(device2, dtype=torch.float)
            clean = clean.to(device2, dtype=torch.float)
           
            if mixture.size(-1) % 16384 != 0:
                  padded_length = 16384 - (mixture.size(-1) % 16384)
                  mixture = torch.cat([mixture, torch.zeros(1, 1,padded_length, device=device2)], dim=-1)

            assert mixture.size(-1) % 16384 == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, 16384, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                    
                    enhanced_chunks.append(self.model(chunk.float()))
            enhanced = torch.cat(enhanced_chunks, dim=-1)
            enhanced = enhanced.to(device2)
            if padded_length != 0:
                    enhanced = enhanced[:, :, :-padded_length]
                    mixture = mixture[:, :, :-padded_length]
                   
           
            front_loss = self.loss_function(clean, enhanced)
           
            model_back.train()
            enhanced = emb(enhanced).double()
            
            
          
            y = model_back(enhanced.float().to(device2))
            back_loss = self.loss_function2(y[0], label[0].to(device2))
            alpha = 0.9
            loss =  ((1-alpha)*back_loss) + (alpha*front_loss) 
            print("Iteration %d in epoch%d--> front_loss = %f back_loss = %f loss = %f " % (i, epoch, front_loss.item() ,back_loss.item(), loss.item()), end='\r')
            loss_total += loss.item()
            front_total += (front_loss.item()*100)
            back_total += back_loss.item()
            
            loss.backward()
            self.optimizer.step()
            optimizer_back.step()
            self.optimizer.zero_grad()
            optimizer_back.zero_grad()
            p = torch.argmax(y[0].detach().cpu(), dim=1)
            intent_p = p
            corr.append((intent_p == label[0]).float())
            if i % 100 ==0:
                   model_back.eval()
                   self.model.eval()
                   for j, (mixture, clean, name, label) in enumerate(self.validation_data_loader):
                       name = name[0]
                       padded_length = 0
                       mixture = mixture.to(device2)
          
                       if mixture.size(-1) % 16384 != 0:
                             padded_length = 16384 - (mixture.size(-1) % 16384)
                             mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device2)], dim=-1)
                       assert mixture.size(-1) % 16384 == 0 and mixture.dim() == 3
                       mixture_chunks = list(torch.split(mixture, 16384, dim=-1))
                       enhanced_chunks = []
                       for chunk in mixture_chunks:
                           enhanced_chunks.append(self.model(chunk.float()).detach().cpu())
                           enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
                           enhanced = enhanced.to(device2)
                       enhanced = emb(enhanced).double()
                       
                       y_pred = model_back(enhanced.float().to(device2))
                       pred = torch.argmax(y_pred[0].detach().cpu(), dim=1)
                       intent_pred = pred
                       correct.append((intent_pred == label[0]).float())
                       if j > 100:
                           break
                   acc = np.mean(np.hstack(correct))
                   intent_acc = acc
                   iter_acc = '\n iteration %d epoch %d -->' %(i, epoch)
                   print(iter_acc, acc, best_accuracy)
                   if intent_acc > best_accuracy:
                       print('Current validation accuracy = {} ({}), updating best model'.format(intent_acc, best_accuracy))
                       best_accuracy = intent_acc
                       best_epoch = epoch
                       torch.save(model_back.state_dict(), '/data/mohamed/jt/code/emb09hitr/best_model.pkl')

        ac = np.mean(np.hstack(corr))
        intent_ac = ac
        iter_ac = '\n iteration %d epoch %d -->' %(i, epoch)
        print(iter_ac, ac, best_acc)
        if intent_ac > best_acc:
            print('Current Training aaccuracy {}, {}'.format(intent_ac, best_acc))
            best_acc = intent_ac
            print('best_acc', best_acc)
            print('improved_acc')
            

        dl_len = len(self.train_data_loader)

        self.writer.add_scalar(f"Total_Train/Loss", loss_total / dl_len, epoch)
        self.writer.add_scalar(f"front_Train/Loss", front_total / dl_len, epoch)
        self.writer.add_scalar(f"back_Train/Loss", back_total / dl_len, epoch)
        self.writer.add_scalar(f"Train/Accuracy", best_acc, epoch)
    @torch.no_grad()
    def _validation_epoch(self, epoch):
            visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
            visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
            visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

            sample_length = self.validation_custom_config["sample_length"]

            stoi_c_n = []  # clean and noisy
            stoi_c_e = []  # clean and enhanced
            pesq_c_n = []
            pesq_c_e = []
            correct = []

            
            for i, (mixture, clean, name, label) in enumerate(self.validation_data_loader):
                name = name[0]
                padded_length = 0

                mixture = mixture.to(self.device)
                clean = clean.to(self.device)

                if mixture.size(-1) % sample_length != 0:
                    padded_length = sample_length - (mixture.size(-1) % sample_length)
                    mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

                assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
                mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

                enhanced_chunks = []
                for chunk in mixture_chunks:
                    enhanced_chunks.append(self.model(chunk.float()).detach().cpu())

                enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
                enhanced = enhanced.to(self.device)
                if padded_length != 0:
                    enhanced = enhanced[:, :, :-padded_length]
                    mixture = mixture[:, :, :-padded_length]
                   
                enhanced = enhanced.cpu().numpy().reshape(-1)
                clean = clean.cpu().numpy().reshape(-1)
                mixture = mixture.cpu().numpy().reshape(-1)

                assert len(mixture) == len(enhanced) == len(clean)
                if i <= visualize_audio_limit:
                    self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                    self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)
                    
                if i <= visualize_waveform_limit:
                    fig, ax = plt.subplots(3, 1)
                    for j, y in enumerate([mixture, enhanced, clean]):
                        ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(np.mean(y), np.std(y), np.max(y), np.min(y)))
                        librosa.display.waveplot(y, sr=16000, ax=ax[j])
                    plt.tight_layout()
                    self.writer.add_figure(f"Waveform/{name}", fig, epoch)
                
                noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
                enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
                clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))
                
                if i <= visualize_spectrogram_limit:
                    fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                    for k, mag in enumerate([noisy_mag, enhanced_mag, clean_mag,]):
                        axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                        librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                    plt.tight_layout()
                    self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)
                
                # Metric
                stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
                stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))




            stoi1_av = np.asarray(stoi_c_n)
            stoi3_av = np.asarray(stoi_c_e)



            stoi1_m = np.max(stoi1_av)
            stoi3_m = np.max(stoi3_av)


            stoi1_a = np.mean(stoi1_av)
            stoi3_a = np.mean(stoi3_av)


            print('The mean c_n_STOI {:.2f}'.format(stoi1_a))
            
            print('The mean c_e_STOI {:.2f}'.format(stoi3_a))

            self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": stoi1_a,
            "Clean and enhanced": stoi3_a
        }, epoch)
            score = (stoi3_a) 
            return score