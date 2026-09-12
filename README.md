# FRDLNet


## Requirements

* python=3.6
* PyTorch=1.2+
* torchvision=0.4.2
* pillow=6.2.1
* numpy=1.18.1
* h5py=1.10.2

## Dataset

#### Fiber

* Change directory to `./filelists/Fiber`
* run `source ./download_Fiber.sh`

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


