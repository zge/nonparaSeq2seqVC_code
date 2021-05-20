#!/bin/bash

# you can set the hparams by using --hparams=xxx
CUDA_VISIBLE_DEVICES=3 python train.py \
  -l logdir \
  -o outdir \
  --n_gpus=1 \
  --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.

# Multi GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py \
  -l logdir \
  -o outdir/vctk/test_wgan_bs16 \
  --n_gpus=2 \
  --hparams=distributed_run=True,batch_size=16,speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.

# Multi GPUs on aipool-linux14
python -m multiproc train.py \
  -l logdir \
  -o outdir/vctk/test_wgan_bs16 \
  -c outdir/vctk/test_wgan_bs32/checkpoint_50000 \
  --n_gpus=4 \
  --hparams=distributed_run=True,batch_size=16,epochs=2000,iters_per_checkpoint=2000,speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.

# Single GPU
CUDA_VISIBLE_DEVICES=1 python train.py \
  -l logdir \
  -o outdir/vctk/test_wgan_bs16 \
  --n_gpus=1 \
  --hparams=distributed_run=False,batch_size=16,speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.

# Single GPU, original features
python train_orig.py \
  -l logdir \
  -o outdir/vctk/test_orig_bs16 \
  -c outdir/vctk/test_orig_bs16/checkpoint_434000 \
  --n_gpus=1 \
  --hparams=distributed_run=False,batch_size=16,epochs=2000,iters_per_checkpoint=5000,speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.
