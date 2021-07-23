# Problem Statement

## [Tiny ImageNet](Tiny_ImageNet/README.md)
- Train ResNet18 on [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) with 70/30 split for 50 Epochs. The target is to achieve 50%+ validation accuracy.

We achieved a `validation accuracy` of `54.45%` and a `train accuracy` of `57.12%` in the `50th epoch`. All the related information can be found [here](Tiny_ImageNet/README.md).

## [COCO K-Means](COCO_K-Means/README.md)
- Download COCO dataset and learn how COCO object detection dataset's schema is.
- Identify following things for this dataset:
  - Class distribution (along with the class names) along with a graph
  - Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.