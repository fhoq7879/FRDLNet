# FRDLNet


## Requirements

* python=3.6
* PyTorch=1.2+
* torchvision=0.4.2
* pillow=6.2.1
* numpy=1.18.1
* h5py=1.10.2

## Dataset

### Dataset Subset for Reproducibility
The representative dataset subset for reproducing core experiments can be downloaded from Release assets:
[FRDLNet‑Dataset.zip](https://github.com/fhoq7879/FRDLNet/releases/download/v1.0-dataset-subset/FRDLNet-Dataset.zip)

File SHA‑256 checksum: `97c80d2daf2e94370bdb07e224254b5a360d99fce8a0294a318980db0baa1758`

> Note: This is only a partial subset of our self‑built cashmere‑wool fiber microscopic dataset for experimental reproducibility.
The full proprietary dataset cannot be fully publicly released. Qualified researchers may obtain the complete dataset upon reasonable formal request to the corresponding author, subject to signing appropriate data‑use agreements.

After downloading `FRDLNet‑Dataset.zip`, unzip it and place the `Fiber` folder under `./filelists/`.

## Train

* method: relationnet|CosineBatch|OurNet.
* n_shot: number of labeled data in each class （1|5）.
* train_aug: perform data augmentation or not during training.
* gpu: gpu id.

```shell
python ./train.py --dataset Fiber  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./train.py --dataset Fiber  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./train.py --dataset Fiber  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```

## Save features

```shell
python ./save_features.py --dataset Fiber  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset Fiber  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./save_features.py --dataset Fiber  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```

## Test

```shell
python ./test.py --dataset Fiber  --model Conv4 --method relationnet --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset Fiber  --model Conv4 --method CosineBatch --n_shot 5 --train_aug --gpu 0
python ./test.py --dataset Fiber  --model Conv4 --method OurNet      --n_shot 5 --train_aug --gpu 0
```


