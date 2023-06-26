# Pruning Transformer via Sparsity

To implement the code in `code/ViT_pruning`, you should follow the steps below. Please note that you can either train the pretrained model and prune on it respectively or just use the pretrained model in 03 CP-ViT (don't forget to change the `pretrained_dir` in python files).

## 00 Dataset

Put CIFAR-10 dataset at `ViT_pruning/data` directory.

```bash
cd ViT_pruning
```

[Download](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.210135637.-1508700061.1687178130) pretrained model `ViT-B_16.npz`.

## 01 P-ViT

### 01-01 Pretrained Model

Run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python P-ViT/train.py
```

You will get a fine-tuned baseline model on CIFAR-10 in `P-ViT/checkpoint` folder.

### 01-02 Evaluation

To evaluate the model with different sparsity, you should first change the `percent` parameter in file `P-ViT/eval.py` Line 54, and then run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python P-ViT/eval.py
```

## 02 AP-ViT

### 02-01 Pretrained Model

Please first put the pretrained model `ViT-B_16.npz` at `AP-ViT/checkpoint` folder, and then run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python AP-ViT/train.py --name $project_name$
```

You will get a fine-tuned clipped model on CIFAR-10 in `AP-ViT/output` folder.

### 02-02 Evaluation

To evaluate the model with different sparsity, you should first change the `prune_ratio` parameter in file `AP-ViT/infer.py`, and then run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python CP-ViT/infer.py --name $project_name$
```

Please note that you should change the python dict `threshold` in file `AP-ViT/infer.py` Line 204 to match the threshold and pruned ratio.

## 03 CP-ViT

### 03-01 Pretrained Model

Please first put the pretrained model `ViT-B_16.npz` at `ViT/Pretrained` folder, and then run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python ViT/train.py
```

You will get a fine-tuned model on CIFAR-10 in `ViT/Model` folder.

### 03-02 Evaluation

To evaluate the model with different sparsity, you should change the `prune_percent` variable in file `CP-ViT/models/modeling.py` by yourself, and then run this code:

```bash
CUDA_VISIBLE_DEVICES=0 python CP-ViT/eval.py
```

