import soundfile as sf
import torch
#import transformers
import fairseq
import glob
import os
import webbrowser
torch.set_num_threads(1)
cp_path = "/data/disk1/data/mnabih/mnabih/time_domain_emb/wav2vec_large.pt"
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path]) 
model = model[0]
model.eval()
model.double()
for i, filename in enumerate(glob.glob('/data/disk1/data/mnabih/mnabih/enhanced_speech_commands_v0.01/wavs/sepeakers/**/*.wav')):
  print(i)
  sig, sr = sf.read(filename)
  sig = (torch.from_numpy(sig).double())
  sig = sig.unsqueeze(dim=0)
  z = model.feature_extractor(sig)
  c = model.feature_aggregator(z)
  torch.save(c, filename.replace('.wav','.pt'))
