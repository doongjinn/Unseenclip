_base_ = './voc12_20_512x512.py'
# dataset settings
data = dict(train=dict(ann_dir=['SegmentationClass', 'SegmentationClassAug'],
                       split=[
                           'ImageSets/Segmentation/train.txt',
                           '/home/dongjin/train_aug.txt'
                       ]))
