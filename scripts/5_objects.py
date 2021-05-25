# The new config inherits a base config to highlight the necessary modification
#_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
_base_ = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5)))
#        mask_head=dict(num_classes=5)))
