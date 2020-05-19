# Introduction

This directory contains PyTorch YOLOv3 software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**.
For more information please visit https://www.ultralytics.com.

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Tutorials

* [GCP Quickstart](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
* [Transfer Learning](https://github.com/ultralytics/yolov3/wiki/Example:-Transfer-Learning)
* [Train Single Image](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Image)
* [Train Single Class](https://github.com/ultralytics/yolov3/wiki/Example:-Train-Single-Class)
* [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)

# Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%

# Inference

`detect.py` runs inference on any sources:

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

To run a specific models:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.weights`  

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights`  

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.weights`  


# Pretrained Weights

Download from: [https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'converted.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'converted.weights'
```

# mAP

```bash
python3 test.py --weights ... --cfg ...
```

- mAP@0.5 run at `--nms-thres 0.5`, mAP@0.5...0.95 run at `--nms-thres 0.7`
- YOLOv3-SPP ultralytics is `ultralytics68.pt` with `yolov3-spp.cfg`
- Darknet results: https://arxiv.org/abs/1804.02767

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |320 |14.0<br>28.7<br>30.5<br>**35.4** |29.1<br>51.8<br>52.3<br>**54.3**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |416 |16.0<br>31.2<br>33.9<br>**39.0** |33.0<br>55.4<br>56.9<br>**59.2**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |512 |16.6<br>32.7<br>35.6<br>**40.3** |34.9<br>57.7<br>59.5<br>**60.6**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**YOLOv3-SPP ultralytics** |608 |16.6<br>33.1<br>37.0<br>**40.9** |35.4<br>58.2<br>60.7<br>**60.9**
