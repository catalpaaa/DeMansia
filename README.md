# DeMansia

## About

DeMansia is a model that integrates ViM with token labeling techniques to enhance performance in image classification tasks.

## Installation

We provided a simple [setup.sh](setup.sh) to install the Conda environment. You need to satisfy the following prerequisite:

- Linux
- NVIDIA GPU
- CUDA 11.8+ supported GPU driver
- Miniforge

Then, simply run `source ./setup.sh` to get started.

## Pretrained Models

These models were trained on the [ImageNet-1k dataset](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) using a single RTX 6000 Ada during our experiments.

Currently, only DeMansia Tiny is available. We will release more models as opportunities arise.

| Name          | Model Dim. | Num. of Layers | Num. of Param. | Input Res. | Top-1  | Top-5  | Batch Size | Download              | Training Log    |
|---------------|------------|----------------|----------------|------------|--------|--------|------------|-----------------------|-----------------|
| DeMansia Tiny | 192        | 24             | 8.06M          | 224²       | 79.37% | 94.51% | 768        | [link][tiny download] | [log][tiny log] |

[tiny download]: https://archive.org/details/DeMansia-Tiny
[tiny log]: https://wandb.ai/catalpa/DeMansia%20Tiny/runs/mec0ihkp

## Training and inferencing

To set up the ImageNet-1k dataset, download both the training and validation sets. Use this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to extract and organize the dataset. You should also download and extract the token labeling dataset from [here](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view?usp=sharing).

We provide [DeMansia train.ipynb](DeMansia%20train.ipynb), which contains all the necessary code to train a DeMansia model and log the training progress. The logged parameters can be modified in [model.py](model.py).

The base model's hyperparameters are stored in [model_config.py](model_config.py), and you can adjust them as needed. When further training our model, note that all hyperparameters are saved directly in the model file. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint). The same applies to inferencing, as PyTorch Lightning automatically handles all parameters when loading our model.

Here's a sample code snippet to perform inferencing with DeMansia:

```python
import torch

from model import DeMansia

model = DeMansia.load_from_checkpoint("path_to.ckpt")
model.eval()

sample = torch.rand(3, 224, 224) # Channel, Width, Height
sample = sample.unsqueeze(0) # Batch, Channel, Width, Height
pred = model(sample) # Batch, # of class
```

## Credits

Our work builds upon the remarkable achievements of [Mamba](https://arxiv.org/abs/2312.00752), [ViM](https://arxiv.org/abs/2401.09417) and [LV-ViT](https://arxiv.org/abs/2104.10858).

[custom_mamba/](custom_mamba) is taken from the [ViM's repo](https://github.com/hustvl/Vim) <3.

[module/data](modules/data) and [module/loss](module/loss) are modified from the [LV-ViT repo](https://github.com/zihangJiang/TokenLabeling).

[module/ema](modules/ema) is modified from [here](https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/utils/__init__.py).
