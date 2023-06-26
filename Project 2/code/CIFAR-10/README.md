## How to run?

Determine the parameters and the model you want to train. Then copy the code to the terminal.

```shell
# All models in './models' except FastResNet9, e.g.: LeNet
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model=LeNet --batch-size=128 --lr=0.1 --max-epoch=200

# FastResNet9
python main_fastresnet9.py --max-epoch=200
```

