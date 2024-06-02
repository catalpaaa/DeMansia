# modified from https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py
# and https://github.com/zihangJiang/TokenLabeling/blob/main/main.py <3

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
from functools import partial
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath, lecun_normal_, trunc_normal_
from torch import Tensor
from torchmetrics.classification import Accuracy

from modules.data import create_token_label_target
from modules.loss import TokenLabelCrossEntropy


class PatchEmbed4_2(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(
        self,
        img_size: tuple[int] = (224, 224),
        patch_size: tuple[int] = (16, 16),
        in_chans=3,
        d_model=768,
    ):
        super().__init__()

        new_patch_size = (patch_size[0] // 2, patch_size[1] // 2)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = d_model

        self.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(
            64, d_model, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


class PatchEmbed4_2_128(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(
        self,
        img_size: tuple[int] = (224, 224),
        patch_size: tuple[int] = (16, 16),
        in_chans=3,
        d_model=768,
    ):
        super().__init__()

        new_patch_size = (patch_size[0] // 2, patch_size[1] // 2)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model

        self.conv1 = nn.Conv2d(
            in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(
            128, d_model, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            conv1=img_size / 2 * img_size / 2 * 3 * 128 * 7 * 7,
            conv2=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            conv3=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            proj=img_size / 2 * img_size / 2 * 128 * self.d_model,
        )
        return sum(block_flops.values())


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_devide_out=if_devide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class DeMansia(pl.LightningModule):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        patch_embd_fn="4_2",
        depth=24,
        d_model=192,
        channels=3,
        num_classes=0,
        learning_rate=5e-4,
        drop_rate=0.0,
        drop_path_rate=0.1,
        initializer_cfg=None,
        init_layer_scale=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm=True,
        residual_in_fp32=True,
        if_abs_pos_embed=True,
        if_token_labeling=False,
        token_label_size=14,
        bimamba_type="v2",
        device=None,
        dtype=None,
        if_devide_out=True,
        ssm_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.d_model = d_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_token_labeling = if_token_labeling
        self.token_label_size = token_label_size
        self.training = False

        if patch_embd_fn == "4_2":
            patch_embd_fn = PatchEmbed4_2
        elif patch_embd_fn == "4_2_128":
            patch_embd_fn = PatchEmbed4_2_128
        else:
            raise ValueError("Set a correct PatchEmbed or I explode you")

        self.patch_embed = patch_embd_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            d_model=d_model,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.patch_embed.num_patches + 1,
                    self.d_model,
                )
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.d_model, num_classes)
        self.aux_head = nn.Linear(self.d_model, num_classes)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.token_loss = TokenLabelCrossEntropy(
            dense_weight=1.0,
            cls_weight=1.0,
            mixup_active=False,
            classes=self.num_classes,
            ground_truth=False,
        )
        self.token_loss.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss()

        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        self.aux_head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.valid_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    # modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def forward_features(
        self,
        x,
        inference_params=None,
    ):
        B, M, _ = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        self.token_position = M // 2
        x = torch.cat(
            (
                x[:, : self.token_position, :],
                cls_token,
                x[:, self.token_position :, :],
            ),
            dim=1,
        )
        M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(
        self,
        x,
        inference_params=None,
    ):
        x = self.patch_embed(x)

        if self.training:
            lam = np.random.beta(1.0, 1.0)
            patch_h, patch_w = x.shape[2], x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_features(
            x,
            inference_params,
        )

        x_cls = self.head(x[:, self.token_position, :])
        if not self.if_token_labeling:
            return x_cls

        x_aux = self.aux_head(
            torch.cat(
                [x[:, : self.token_position, :], x[:, self.token_position + 1 :, :]],
                dim=1,
            )
        )

        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]

        x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
        temp_x = x_aux.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
        x_aux = temp_x
        x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def training_step(self, batch):
        self.training = True

        sample, target = batch
        preds = self(sample)

        if self.if_token_labeling:
            target = create_token_label_target(
                target,
                num_classes=self.num_classes,
                smoothing=0.1,
                device=self.device,
                label_size=self.token_label_size,
            )
            loss = self.token_loss(preds, target)
        else:
            loss = self.ce_loss(preds, target)

        self.log("Training Loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.training = False
        return loss

    def validation_step(self, batch):
        sample, target = batch
        preds = self(sample)

        loss = self.ce_loss(preds, target)

        self.log(
            "Validation Accuracy Top 1",
            self.valid_acc_top_1(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation Accuracy Top 5",
            self.valid_acc_top_5(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("Validation Loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        pl.utilities.memory.garbage_collection_cuda()

    def on_validation_epoch_end(self):
        pl.utilities.memory.garbage_collection_cuda()

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05,
            decoupled_weight_decay=True,
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "Learning Rate",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
