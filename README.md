## A PyTorch implementation of Graph Neural Network
This is a collection of PyTorch implementation of Graph Neural Networks.
* Graph Convolutional Networks
* Gated Graph Sequence Neural Networks
* Relational Graph Convolutional Networks

## Environment
All codes are run on a machine with 2.3 GHz 8-Core Intel Core i9 with 32 GB RAM. 
* python 3.8.5
* pytorch 1.6.0

## Graph Convolutional Networks [[paper]](https://arxiv.org/abs/1609.02907)
The code was adapted from [here](https://github.com/tkipf/pygcn).
### Usage
Dataset(cora, citeseer, WebKB) is automatically downloaded by runing [main.py](./gcn/main.py).
```
cd gcn
python main.py --dataset cora
python main.py --dataset citeseer
python main.py --dataset WebKB --sub_dataset cornell
python main.py --dataset WebKB --sub_dataset texas
python main.py --dataset WebKB --sub_dataset washington
python main.py --dataset WebKB --sub_dataset wisconsin
```

## Gated Graph Sequence Neural Networks [[paper]](https://arxiv.org/abs/1511.05493)
The code was adapted from [here](https://github.com/chingyaoc/ggnn.pytorch).
### Usage
Download bAbI dataset by running [getdata.sh](./ggnn/getdata.sh). Before running it, you have to install [torch](http://torch.ch/docs/getting-started.html#_).
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
bash install-deps
./install.sh
```
Now, get back to the project for training ggnn:
```
cd ggnn
python main.py --verbal --task_id 4 --state_dim 4 --n_epochs 10
python main.py --verbal --task_id 15 --state_dim 5 --n_epochs 10
python main.py --verbal --task_id 16 --state_dim 10 --n_epochs 150
```
## Relational Graph Convolutional Networks [[paper]](https://arxiv.org/abs/1703.06103)
The code was adapted from [here](https://github.com/mjDelta/relation-gcn-pytorch).
Before running the code, you have to download datasets (aifb, mutag, am).
```
cd rgcn
python prepare_dataset.py --dataset aifb
python prepare_dataset.py --dataset mutag
python prepare_dataset.py --dataset am
```
Then, run [main.py](./rgcn/main.py) to train rgcn model.
```
python main.py --dataset aifb
python main.py --dataset mutag
python main.py --dataset am
```

