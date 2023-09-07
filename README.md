# TRT YOLOV7
## Table of contents
- Introduction
- Prerequires
- Installation 
- How to use 
- Todo


## INTRODUCTION
With the original yolov7, after exporting to onnx and finally tensorrt, there is currently a bug at the confidence score, giving the wrong confidence score of the object. Thanks to pinto's yolov7 customization, thus correcting the incorrect confidence score


## PREREQUIRES
- Ubuntu 18.04
- Tensorrt 8.5.2.2


## HOW TO USE
1. Download yolov7 256x320 model from [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7)
2. Use trt exec to convert from onnx model to tensorrt engine
```
/usr/src/tensorrt/bin/trtexec --onnx=src_model.onnx --saveEngine=dst_model.trt --fp16 --explicitBatch --minShapes=images:1x3x256x320 --maxShapes=images:4x3x256x320 --optShapes=images:2x3x256x320 --workspace=2048
```
3. Place model to app/model + add image folder to run.py. Then,
```
python3 run.py
```

REFERENCE:
[307_YOLOv7](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7/demo)
