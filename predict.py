import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--outdir', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--view_img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, view_img = opt.outdir, opt.source, opt.weights, opt.view_img

    device = torch_utils.select_device(opt.device)
    if os.path.exists(opt.outdir):
        shutil.rmtree(opt.outdir)
    os.makedirs(opt.outdir)

    model = Darknet(opt.cfg, img_size, trace=False)

    if weights.endswith('.pt'):  # load pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # load darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    dataset = LoadImages(source, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        t_start = time.time()
        img = torch.from_numpy(img).to(device).unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        t_elapse = time.time() - t_start

        save_path = str(Path(out) / Path(path).name)
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

            for *xyxy, conf, _, cls in det:
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('%sDone. (%.3fs)' % (s, t_elapse))

        cv2.imwrite(save_path, im0)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    with torch.no_grad():
        main()
