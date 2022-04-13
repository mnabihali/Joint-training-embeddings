# Joint training embeddings
---
This is the official implementation of the submitted Interspeech 2022 paper [Enhancing Embeddings for Speech Classification in Noisy Conditions], (Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti). 

## Description
---
In this paper, we investigate how enhancement can be applied in neural speech classification architectures employing pre-trained speech embeddings. We investigate two approaches: one applies time-domain enhancement prior to extracting the embeddings; the other employs a convolutional neural network to map the noisy embeddings to the corresponding clean ones. All the experiments are conducted based on Fluent Speech commands, Google Speech commands v0.01, and generated noisy versions of these datasets.



## Steps of Embeds-Enh Strategy

### Embeddings Extraction
---
You need to download the wav2vec model from (https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) to extract the embeddings from the dataset.
```bash
python emb.py
```

### Trainining
Use `train.py` to jointly train both the speech enhancement and the speech classifier modules. It receivers six commands line parameters:
- `-m` path to save the best back-end model.
- `-M` path to save the best front-end model.
- `-b` number of residual blocks in the back-end speech classifier.
- `-r` number of repeats of the residual blocks.
- `-lr` learning rate
- `e` number of epochs
