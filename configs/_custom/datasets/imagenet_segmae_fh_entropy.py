# dataset settings
dataset_type = 'ImageNet'
data_root = '/media/data1/wjy/dataset/ImageNet1k/train/'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFHSegMap'),
    dict(
        type='SegRandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5), # plan to give up
    dict(type='PackInputs', meta_keys=['img_path','gt_seg_mask', 'num_of_objects', 'unique_part', 'entropy'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    # 训练数据集配置
    dataset=dict(
        type='SegMAEImageList',
        ann_file='/media/data2/imagenet_train_FH_500_500_entropy',
        ann_root='/media/data2/imagenet_train_FH_500_500_entropy',
        data_prefix=data_root,
        pipeline=train_pipeline
    )
)


