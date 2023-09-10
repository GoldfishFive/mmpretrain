_base_ = [
    '../datasets/imagenet_segmae_fh_entropy.py',
    # '../datasets/val_imagenet_segmae_fh_entropy.py',
    '../default_runtime.py',
]
# model settingsm
model = dict(
    type='SegMAE',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='SegMAEVit_VR',
        arch='b',
        patch_size=16,
        mask_ratio=0.75,
        fix_mask_ratio=False,# True used the fixed mask_ratio 0.75 during training;
        max_epochs=5, # when fix_mask_ratio is False, mask_ratio change from low_mask_ratio to high_mask_ratio
        low_mask_ratio=0.35,
        high_mask_ratio=0.75
    ),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        pad_with_cls_token=True
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', distribution='uniform', layer='Linear'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# dataset 8 x 128
batch_size_pre_GPU = 32
GPU_NUMBER= 1
train_dataloader = dict(batch_size=batch_size_pre_GPU, num_workers=8)
total_batch = batch_size_pre_GPU * GPU_NUMBER
# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * total_batch / 256, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        by_epoch=True,
        begin=1,
        end=5,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 300 epochs
train_cfg = dict(type='EpochBasedTrainLoop',max_epochs=5)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='SetEpochHook',
        start_epoch=0
)]
# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
