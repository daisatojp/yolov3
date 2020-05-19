import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import utils.torch_utils as torch_utils
from utils.utils import init_seeds, labels_to_class_weights, compute_loss
from utils.datasets import LoadImagesAndLabels
from models import parse_data_cfg, parse_model_cfg, Darknet
import test

# Hyperparameters
# results68: 59.2 mAP@0.5 yolov3-spp-416
# see https://github.com/ultralytics/yolov3/issues/310
hyp = {
    'giou': 3.54,  # giou loss gain
    'cls': 37.4,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 64.3,  # obj loss gain (*=img_size/416)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.225,  # iou training threshold
    'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
    'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.000484,  # optimizer weight decay
    'fl_gamma': 0.5,  # focal loss gamma
    'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
    'degrees': 1.98,  # image rotation (+/- deg)
    'translate': 0.05,  # image translation (+/- fraction)
    'scale': 0.05,  # image scale (+/- gain)
    'shear': 0.641  # image shear (+/- deg)
}


# sparsity-induced penalty term
# x_{k+1} = x_{k} - \alpha_{k} * g^{k}
def update_bn(scale, model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(scale * torch.sign(m.weight.data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epoch_num', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_num', type=int, default=4)
    parser.add_argument('--cfg', type=str, default=os.path.join('cfg', 'yolov3-spp.cfg'))
    parser.add_argument('--data', type=str, default=os.path.join('data', 'coco2017.data'))
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--weight', type=str, default=None,
                        help='initial weight path')
    parser.add_argument('--multi-scale', action='store_true',
                        help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache_images', action='store_true',
                        help='cache images for faster training')
    parser.add_argument('--arc', type=str, default='default',
                        help='yolo architecture')  # default pw, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--adam', action='store_true',
                        help='use adam optimizer')
    parser.add_argument('--var', type=float,
                        help='debug variable')
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='enable sparsity training (recommend: 0.0001)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='mixed precision training https://github.com/NVIDIA/apex')
    parser.add_argument('--outdir', type=str,
                        help='output directory')
    args = parser.parse_args()

    print(args)

    device = args.device
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    accumulate_num = args.accumulate_num
    cfg_path = args.cfg
    data_path = args.data
    img_size = args.img_size
    weight_path = args.weight
    mixed_precision = args.mixed_precision
    out_dir = args.outdir

    try:
        from apex import amp
    except ModuleNotFoundError as e:
        mixed_precision = False

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    hyp['obj'] *= img_size / 416

    tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb'))

    if 'pw' not in args.arc:  # remove BCELoss positive weights
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    init_seeds()

    img_sz_min = img_size
    img_sz_max = img_size
    if args.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('using multi-scale {} - {}'.format(img_sz_min * 32, img_sz_max * 32))

    # Configure run
    data_dict = parse_data_cfg(data_path)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    class_num = int(data_dict['classes'])

    model = Darknet(cfg_path, arc=args.arc).to(device)

    parameter_group_0 = []
    parameter_group_1 = []
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            parameter_group_1 += [v]
        else:
            parameter_group_0 += [v]
    if args.adam:
        optimizer = optim.Adam(parameter_group_0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(parameter_group_0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': parameter_group_1, 'weight_decay': hyp['weight_decay']})
    del parameter_group_0, parameter_group_1

    start_epoch = 0
    best_fitness = float('inf')
    if weight_path is not None:
        chkpt = torch.load(weight_path, map_location=device)
        if 'model' in chkpt:
            model.load_state_dict(chkpt, strict=False)
        if 'optimizer' in chkpt:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
        if 'epoch' in chkpt:
            start_epoch = chkpt['epoch'] + 1

    scheduler = lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[round(epoch_num * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('using {} dataloader workers'.format(num_workers))

    dataset_train = LoadImagesAndLabels(
        path=train_path,
        img_size=img_size,
        batch_size=batch_size,
        augment=True,
        hyp=hyp,  # augmentation hyperparameters
        rect=args.rect,  # rectangular training
        image_weights=False,
        cache_labels=epoch_num > 10,
        cache_images=args.cache_images)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not args.rect,
        pin_memory=True,
        collate_fn=dataset_train.collate_fn)

    dataset_test = LoadImagesAndLabels(
        path=test_path,
        img_size=img_size,
        batch_size=batch_size,
        hyp=hyp,
        rect=True,
        cache_labels=True,
        cache_images=args.cache_images)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset_test.collate_fn)

    model.nc = class_num
    model.arc = args.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset_train.labels, class_num).to(device)  # attach class weights

    Bi = 0
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    for epoch in range(start_epoch, epoch_num):
        model.train()
        mean_loss = []
        mean_loss_box = []
        mean_loss_obj = []
        mean_loss_cls = []
        mean_small_layer_num = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, (imgs, targets, paths, _) in progress_bar:
            Bi += 1
            imgs = imgs.type(torch.float32).to(device) / 255.0
            targets = targets.to(device)

            if args.multi_scale:
                if ni / accumulate_num % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            pred = model(imgs)

            loss, loss_box, loss_obj, loss_cls = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('loss({}) is non-finite'.format(loss_items))
                exit(-1)

            mean_loss.append(loss.item())
            mean_loss_box.append(loss_box.item())
            mean_loss_obj.append(loss_obj.item())
            mean_loss_cls.append(loss_cls.item())

            loss = loss / accumulate_num

            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if Bi % accumulate_num == 0:
                if 0.0 < args.sparsity:
                    update_bn(args.sparsity, model)
                optimizer.step()
                optimizer.zero_grad()

            small_layer_num = 0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    small_layer_num += np.count_nonzero(
                        m.weight.data.abs().detach().cpu().numpy() < 0.1)
            mean_small_layer_num.append(small_layer_num)

        scheduler.step()

        final_epoch = (epoch + 1 == epoch_num)

        mean_loss = np.mean(mean_loss)
        mean_loss_box = np.mean(mean_loss_box)
        mean_loss_obj = np.mean(mean_loss_obj)
        mean_loss_cls = np.mean(mean_lsos_cls)

        with torch.no_grad():
            results, maps = test.test(
                device=device,
                cfg=cfg_path,
                data=data_path,
                batch_size=batch_size,
                img_size=img_size,
                model=model,
                conf_thres=0.1,
                save_json=False,
                dataloader=dataloader_test)

        titles = [
            'train_loss',
            'train_loss_box',
            'train_loss_obj',
            'train_loss_cls',
            'P', 'R', 'mAP', 'F1',
            'val_loss_box',
            'val_loss_obj',
            'val_loss_cls',
            'small_layer_num'
        ]
        values = [
            mean_loss,
            mean_loss_box,
            mean_loss_obj,
            mean_loss_cls,
            results[0],
            results[1],
            results[2],
            results[3],
            results[4],
            results[5],
            results[6],
            mean_small_layer_num
        ]
        for title, value in zip(titles, values):
            print('{}={}'.format(title, value))
            tb_writer.add_scalar(title, xi, epoch)

        fitness = sum(results[4:])  # total loss
        if fitness < best_fitness:
            best_fitness = fitness

        chkpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'model': (model.module.state_dict()
                      if type(model) is nn.parallel.DistributedDataParallel
                      else model.state_dict()),
            'optimizer': None if final_epoch else optimizer.state_dict()
        }
        torch.save(chkpt, os.path.join(out_dir, 'last.pt'))
        if best_fitness == fitness:
            torch.save(chkpt, os.path.join(out_dir, 'best.pt'))
        if (epoch + 1) % 10 == 0:
            torch.save(chkpt, os.path.join(out_dir, 'backup_{}.pt'.format(epoch)))
        del chkpt


if __name__ == '__main__':
    main()
