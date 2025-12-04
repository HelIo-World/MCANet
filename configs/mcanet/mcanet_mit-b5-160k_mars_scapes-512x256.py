_base_ = [
    '../_base_/models/mcanet_mit-b5.py', '../_base_/datasets/mars_scapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

size = (512, 256)
data_preprocessor = dict(size=size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=8,
        ignore_index=255
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optim_wrapper = dict(
    type='OptimWrapper',
    _delete_=True,
    optimizer=dict(
        type='SGD', 
        lr=0.005,  # 临时降到 0.005，或者更低 0.001
        momentum=0.9, 
        weight_decay=0.0005
    ),
    clip_grad=dict(max_norm=5, norm_type=2) 
)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=4000)