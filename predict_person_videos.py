import os
import time
import argparse
import random
from glob import glob
from tqdm import tqdm
import cv2
import torch
from models import Darknet
from utils.datasets import LoadImages
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
from utils.parse_config import parse_data_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--cfg', type=str, default=os.path.join('cfg', 'yolov3-spp.cfg'))
    parser.add_argument('--data', type=str, default=os.path.join('data', 'coco2014.data'))
    parser.add_argument('--weights', type=str, default=os.path.join('weights', 'yolov3-spp-ultralytics.pt'))
    parser.add_argument('--img_size', type=int, default=640, help='inference size')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    device = opt.device
    src_dir = opt.src
    dst_dir = opt.dst
    cfg_path = opt.cfg
    data_path = opt.data
    weights_path = opt.weights
    img_size = opt.img_size
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres

    src_paths = glob(os.path.join(src_dir, '*.AVI'))

    model = Darknet(opt.cfg, img_size, trace=False)

    if weights_path.endswith('.pt'):  # load pytorch format
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    else:  # load darknet format
        _ = load_darknet_weights(model, weights_path)

    model.to(device).eval()

    for src_path in src_paths:
        src_name = os.path.splitext(os.path.basename(src_path))[0]
        out_dir = os.path.join(dst_dir, src_name)

        if os.path.exists(out_dir):
            continue

        print('out_dir={}'.format(out_dir))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dataset = LoadImages(src_path, img_size=img_size)

        # Get classes and colors
        classes = load_classes(parse_data_cfg(data_path)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        for i, (path, img, im0s, vid_cap) in enumerate(tqdm(dataset), start=1):

            img = torch.from_numpy(img).to(device).unsqueeze(0)
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, nms_thres)

            save_path = os.path.join(out_dir, '{:08d}.jpg'.format(i))
            s = ''
            det = pred[0]
            im0 = im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                person_detected = False
                for *xyxy, conf, _, cls in det:
                    if int(cls) == 0 and (1000 < float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))):
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        person_detected = True

                if person_detected:
                    cv2.imwrite(save_path, im0)


if __name__ == '__main__':
    with torch.no_grad():
        main()
