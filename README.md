# Joint training embeddings
---
This is the official implementation of the submitted Interspeech 2022 paper [Enhancing Embeddings for Speech Classification in Noisy Conditions], (Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti). 

## Description
---
In this paper, we investigate how enhancement can be applied in neural speech classification architectures employing pre-trained speech embeddings. We investigate two approaches: one applies time-domain enhancement prior to extracting the embeddings; the other employs a convolutional neural network to map the noisy embeddings to the corresponding clean ones. All the experiments are conducted based on Fluent Speech commands, Google Speech commands v0.01, and generated noisy versions of these datasets.

## Architectures
<img src="https://github.com/mnabihali/Joint-training-embeddings/blob/main/assets/systems.png" width="512"/>

## Steps of Wave-Enh Strategy
You need to download the wav2vec model from (https://github.com/pytorch/fairseq/tree/main/examples/wav2vec), and modigy its path in `util/utils.py` file.

### Trainining
Use `train.py` to jointly train both the speech enhancement and the speech classifier modules. It receives two command line parameters:
- `-C, --config`, the path of your configuration file for training process.
- `-R, --resume`, resume training from the last saved checkpoint.

Syntax `python train.py -C config/train/train.json` or `python train.py config/train/train.json -R`

### Evaluation
Use `enhancement.py` to evaluate both models. 

Syntax: `python enhancement.py -C config/enhancement/unet_basic.json -D 0 -O <path to save the enhanced signals> -M <path to the speech enhancement model> -m <path to the back end speech classifier>

## Steps of Embeds-Enh Strategy

### Embeddings Extraction
---
You need to download the wav2vec model from (https://github.com/pytorch/fairseq/tree/main/examples/wav2vec), and modify its path in `emb.py` file to extract the embeddings from the dataset.

Syntax: `python emb.py`

### Trainining
Use `train.py` to jointly train both the speech enhancement and the speech classifier modules. It receives six main commands line parameters:
- `-m` path to save the best back-end model.
- `-M` path to save the best front-end model.
- `-b` number of residual blocks in the back-end speech classifier.
- `-r` number of repeats of the residual blocks.
- `-lr` learning rate
- `e` number of epochs

Syntax: `python train.py -m ./best_bkmodel.pkl -M ./best_frmodel.pkl -b 5 -r 2 -lr 0.001 -e 100`

### Evaluation
Use `evaluation.py` to evaluate both models based on the test dataset. It receives two main commands line parameters:
- `-m` path to save the best back-end model.
- `-M` path to save the best front-end model.

Syntax: `python evaluation.py -m ./best_bkmodel.pkl -M ./best_frmodel.pkl `
