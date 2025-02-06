import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/WIDERFace'


def train(args, params):
    # Model
    model = nn.version_n(params)
    model.cuda()

    # Optimizer
    accumulate = max(round(32 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 32

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    for filename in os.listdir(f'{data_dir}/images/train'):
        filenames.append(f'{data_dir}/images/train/' + filename)

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    step = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(args, params)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'kpt',
                                                     'Recall', 'Precision', 'mAP@50', 'F1', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'kpt'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_kpt_loss = util.AverageMeter()

            for samples, targets in p_bar:

                samples = samples.cuda()
                scheduler.step(step, optimizer)

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_cls, loss_box, loss_kpt = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_kpt_loss.update(loss_kpt.item(), samples.size(0))

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_kpt).backward()

                # Optimize
                if step % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_kpt_loss.avg)
                    p_bar.set_description(s)

                step += 1

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'kpt': str(f'{avg_kpt_loss.avg:.3f}'),
                                 'F1': str(f'{last[0]:.3f}'),
                                 'mAP': str(f'{last[1]:.3f}'),
                                 'mAP@50': str(f'{last[2]:.3f}'),
                                 'Recall': str(f'{last[3]:.3f}'),
                                 'Precision': str(f'{last[4]:.3f}')})
                log.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[1]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.disable_grad('./weights/best.pt')  # strip optimizers
        util.disable_grad('./weights/last.pt')  # strip optimizers


def test(args, params, model=None):
    filenames = []
    for filename in os.listdir(f'{data_dir}/images/val'):
        filenames.append(f'{data_dir}/images/val/' + filename)

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float()

    model.eval()

    nms = util.NMS(params)

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    f1 = 0
    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'P', 'R', 'F1', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        # Inference
        with torch.no_grad():
            outputs = model(samples)
        # NMS
        outputs = nms(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            output = output[0]
            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls.reshape(-1, 1), box), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, f1, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, f1, mean_ap))
    # Return results
    model.float()  # for training
    return f1, mean_ap, map50, m_rec, m_pre


def demo(args, params):
    import cv2
    import numpy
    model = torch.load(f='./weights/best.pt', map_location='cuda')
    model = model['model'].float().fuse()
    model.half()
    model.eval()

    nms = util.NMS(params, conf_threshold=0.325)

    camera = cv2.VideoCapture('./data/video.mp4')
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, args.input_size / shape[0], args.input_size / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = numpy.mod(args.input_size - pad[0], 32) / 2
            h = numpy.mod(args.input_size - pad[1], 32) / 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image,
                                   dsize=pad,
                                   interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image,
                                       top, bottom,
                                       left, right,
                                       cv2.BORDER_CONSTANT)  # add border
            # Convert HWC to CHW, BGR to RGB
            image = image.transpose((2, 0, 1))[::-1]
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image.half()
            image = image.cuda()
            # Inference
            with torch.no_grad():
                outputs = model(image)
            # NMS
            outputs = nms(outputs)
            for output in outputs:

                box_output = output[0]
                kpt_output = output[1]
                if len(box_output) == 0 or len(kpt_output) == 0:
                    continue
                kpt_output = kpt_output.reshape((-1, 5, 2))
                r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

                box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                box_output[:, :4] /= r

                box_output[:, 0].clamp_(0, shape[1])  # x
                box_output[:, 1].clamp_(0, shape[0])  # y
                box_output[:, 2].clamp_(0, shape[1])  # x
                box_output[:, 3].clamp_(0, shape[0])  # y

                kpt_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                kpt_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                kpt_output[..., 0] /= r
                kpt_output[..., 1] /= r
                kpt_output[..., 0].clamp_(0, shape[1])  # x
                kpt_output[..., 1].clamp_(0, shape[0])  # y

                box_output = box_output.cpu().numpy()
                kpt_output = kpt_output.cpu().numpy()

                for box, kpt in zip(box_output, kpt_output):
                    x1, y1, x2, y2 = list(map(int, box[:4]))
                    cv2.rectangle(frame,
                                  pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3)
                    for i in kpt:
                        cv2.circle(frame,
                                   center=(int(i[0]), int(i[1])),
                                   radius=3, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            cv2.imshow(winname='Frame', mat=frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # Break the loop
        else:
            break
    # Closes all the frames
    cv2.destroyAllWindows()


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.version_n(params).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)
    if args.demo:
        demo(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
