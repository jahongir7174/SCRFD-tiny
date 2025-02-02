import copy
import math
import random

import numpy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def export():
    model = torch.load(f='./weights/best.pt', map_location='cpu')
    model = model['model'].float().fuse()

    ts = torch.jit.trace(model, torch.randn((1, 3, 640, 640)), strict=False)
    ts.save('./weights/best.model')

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ['input.1']
    output_names = ['score_8', 'score_16', 'score_32',
                    'bbox_8', 'bbox_16', 'bbox_32', ]

    # If model graph contains keypoints strides add keypoints to outputs
    output_names += ['kps_8', 'kps_16', 'kps_32']

    # Define dynamic axes for export
    dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
    dynamic_axes[input_names[0]] = {
        0: '?',
        2: '?',
        3: '?'
    }

    torch.onnx.export(model,
                      torch.randn((1, 3, 640, 640)),
                      './weights/last.onnx',
                      keep_initializers_as_inputs=False,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(tp, conf, output, target, plot=False, names=(), eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    if plot:
        names = dict(enumerate(names))  # to dict
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        plot_pr_curve(px, py, ap, names, save_dir="./weights/PR_curve.png")
        plot_curve(px, f1, names, save_dir="./weights/F1_curve.png", y_label="F1")
        plot_curve(px, p, names, save_dir="./weights/P_curve.png", y_label="Precision")
        plot_curve(px, r, names, save_dir="./weights/R_curve.png", y_label="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    f1 = f1.mean()
    m_pre = p.mean()
    m_rec = r.mean()
    map50 = ap50.mean()
    mean_ap = ap.mean()
    return tp, fp, m_pre, m_rec, f1, map50, mean_ap


def disable_grad(filename):
    x = torch.load(filename, map_location="cpu")
    for p in x['model'].parameters():
        p.requires_grad_ = False
    torch.save(x, f=filename)


def clip_gradients(model, max_norm=10):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 1500))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 1500))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class AnchorGenerator:
    def __init__(self, strides, ratios, scales, sizes):

        # calculate sizes of anchors
        self.strides = [(s, s) for s in strides]

        self.ratios = torch.Tensor(ratios)
        self.scales = torch.Tensor(scales)

        self.anchors = []
        for size in sizes:
            w = size
            h = size
            x_center = 0.0
            y_center = 0.0
            h_ratios = torch.sqrt(self.ratios)
            w_ratios = 1 / h_ratios

            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)

            # use float anchor and the anchor's center is aligned with the pixel center
            self.anchors.append(torch.stack(tensors=[x_center - 0.5 * ws,
                                                     y_center - 0.5 * hs,
                                                     x_center + 0.5 * ws,
                                                     y_center + 0.5 * hs], dim=-1))

    def make_anchors(self, sizes, device='cuda'):
        anchors = []
        for i in range(len(self.strides)):
            h, w = sizes[i]
            h, w = int(h), int(w)
            shift_x = torch.arange(0, w, device=device) * self.strides[i][0]
            shift_y = torch.arange(0, h, device=device) * self.strides[i][1]

            shift_x, shift_y = self._meshgrid(shift_x, shift_y)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)
            shifts = shifts.type_as(self.anchors[i].to(device))
            anchors.append((self.anchors[i].to(device)[None, :, :] + shifts[:, None, :]).view(-1, 4))
        return anchors

    def valid_flags(self, sizes, shape, device='cuda'):
        flags = []
        for i in range(len(self.strides)):
            h = min(int(numpy.ceil(shape[0] / self.strides[i][1])), sizes[i][0])
            w = min(int(numpy.ceil(shape[1] / self.strides[i][0])), sizes[i][1])

            valid_x = torch.zeros(sizes[i][1], dtype=torch.bool, device=device)
            valid_y = torch.zeros(sizes[i][0], dtype=torch.bool, device=device)
            valid_x[:w] = 1
            valid_y[:h] = 1
            valid_x, valid_y = self._meshgrid(valid_x, valid_y)
            valid = valid_x & valid_y
            flags.append(valid[:, None].expand(valid.size(0), self.anchors[i].size(0)).contiguous().view(-1))
        return flags

    @staticmethod
    def _meshgrid(x, y):
        shift_x = x.repeat(len(y))
        shift_y = y.view(-1, 1).repeat(1, len(x)).view(-1)
        return shift_x, shift_y


class ATSSAssigner:
    INF = 100000000

    def __init__(self, top_k=9):
        self.top_k = top_k

    def __call__(self,
                 bboxes,
                 num_level_bboxes,
                 gt_bboxes):
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.compute_iou(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        gt_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_area = torch.sqrt(torch.clamp(gt_width * gt_height, min=1e-4))

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]  # (A,G)
            selectable_k = min(self.top_k, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # candidate anchors (topk*num_level_bboxes, G) = (AK, G)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]  # (AK,G)
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        dist_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0]  # (A,G)
        dist_min.div_(gt_area)

        is_pos = is_pos & (dist_min > 0.001)

        # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps, -self.INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -self.INF] = argmax_overlaps[max_overlaps != -self.INF] + 1
        return assigned_gt_inds

    @staticmethod
    def compute_iou(boxes1, boxes2):
        assert boxes1.size(-1) in [0, 4, 5]
        assert boxes2.size(-1) in [0, 4, 5]
        if boxes2.size(-1) == 5:
            boxes2 = boxes2[..., :4]
        if boxes1.size(-1) == 5:
            boxes1 = boxes1[..., :4]
        assert (boxes1.size(-1) == 4 or boxes1.size(0) == 0)
        assert (boxes2.size(-1) == 4 or boxes2.size(0) == 0)

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert boxes1.shape[:-2] == boxes2.shape[:-2]
        batch_shape = boxes1.shape[:-2]

        rows = boxes1.size(-2)
        cols = boxes2.size(-2)

        if rows * cols == 0:
            return boxes1.new(batch_shape + (rows, cols))

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (
                boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (
                boxes2[..., 3] - boxes2[..., 1])

        lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        union = torch.max(union, union.new_tensor([1e-6]))
        return overlap / union


class QualityFocalLoss(torch.nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, output, target, weight=None, avg_factor=None):
        # label denotes the category id, score denotes the quality score
        label, score = target

        # negatives are supervised by 0 quality score
        output_sig = output.sigmoid()
        scale_factor = output_sig
        zero_label = scale_factor.new_zeros(output.shape)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output,
                                                                    zero_label,
                                                                    reduction='none') * scale_factor.pow(self.beta)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos = ((label >= 0) & (label < output.size(1))).nonzero(as_tuple=False).squeeze(1)
        pos_label = label[pos].long()
        # positives are supervised by bbox quality (IoU) score
        scale_factor = score[pos] - output_sig[pos, pos_label]
        loss[pos, pos_label] = torch.nn.functional.binary_cross_entropy_with_logits(
            output[pos, pos_label],
            score[pos],
            reduction='none') * scale_factor.abs().pow(self.beta)

        loss = loss.sum(dim=1, keepdim=False)
        if weight is not None:
            loss = loss * weight
        return loss.sum() / avg_factor


class DIoULoss(torch.nn.Module):

    def __init__(self, eps=1e-6, loss_weight=2.0):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, output, target, weight, avg_factor):
        if weight is not None and not torch.any(weight > 0):
            return (output * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == output.shape
            weight = weight.mean(-1)

        lt = torch.max(output[:, :2], target[:, :2])
        rb = torch.min(output[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (output[:, 2] - output[:, 0]) * (output[:, 3] - output[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + self.eps

        # enclose area
        enclose_x1y1 = torch.min(output[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(output[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw ** 2 + ch ** 2 + self.eps

        b1_x1, b1_y1 = output[:, 0], output[:, 1]
        b1_x2, b1_y2 = output[:, 2], output[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        rho2 = left + right

        # DIoU
        loss = 1 - (overlap / union - rho2 / c2)
        if weight is not None:
            loss = loss * weight
        loss = loss.sum() / avg_factor
        return self.loss_weight * loss


class SmoothL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1 / 9.0
        self.loss_weight = 0.1

    def forward(self, output, target, weight, avg_factor):
        diff = torch.abs(output - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        return self.loss_weight * (loss * weight).sum() / avg_factor


def distance2box(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def distance2kpt(points, distance, max_shape=None):
    output = []
    for i in range(0, distance.shape[1], 2):
        x = points[:, i % 2 + 0] + distance[:, i]
        y = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            x = x.clamp(min=0, max=max_shape[1])
            y = y.clamp(min=0, max=max_shape[0])
        output.append(x)
        output.append(y)
    return torch.stack(output, -1)


class ComputeLoss:
    def __init__(self, args, params):
        self.nc = 1
        self.nk = 5
        self.args = args

        self.loss_box = DIoULoss()
        self.loss_kpt = SmoothL1Loss()
        self.loss_cls = QualityFocalLoss()
        self.assigner = ATSSAssigner()
        self.anchor_generator = AnchorGenerator(params['face_anchors']['strides'],
                                                params['face_anchors']['ratios'],
                                                params['face_anchors']['scales'],
                                                params['face_anchors']['sizes'])

    def __call__(self, outputs, targets):
        x_cls, x_box, x_kpt = outputs
        sizes = [i.size()[-2:] for i in x_cls]
        device = x_cls[0].device
        n = x_cls[0].shape[0]
        anchors = self.anchor_generator.make_anchors(sizes, device)
        anchors = [anchors for _ in range(n)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        shape = [self.args.input_size, self.args.input_size]
        for _ in range(n):
            multi_level_flags = self.anchor_generator.valid_flags(sizes, shape, device)
            valid_flag_list.append(multi_level_flags)

        y_cls = []
        y_box = []
        y_kpt = []
        for i in range(n):
            idx = targets['idx'] == i
            y_cls.append(targets['cls'][idx].to(x_cls[0].device).to(torch.long))
            y_box.append(targets['box'][idx].to(x_box[0].device).to(x_box[0].dtype))
            y_kpt.append(targets['kpt'][idx].to(x_kpt[0].device).to(x_kpt[0].dtype))
        targets = self.get_targets(anchors, valid_flag_list, y_box, y_kpt, y_cls, n)
        if targets is None:
            return None

        (y_anchors,
         y_labels, y_label_weights,
         y_box_targets, y_box_weights,
         y_kpt_targets, y_kpt_weights,
         num_total_pos, num_total_neg) = targets

        num_total_samples = self.reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls = []
        losses_box = []
        losses_kpt = []
        avg_factor = []
        for i in range(len(y_anchors)):
            stride = self.anchor_generator.strides[i]
            anchor = y_anchors[i].reshape(-1, 4)

            box = x_box[i].permute(0, 2, 3, 1).reshape(-1, 4)
            cls = x_cls[i].permute(0, 2, 3, 1).reshape(-1, self.nc)
            kpt = x_kpt[i].permute(0, 2, 3, 1).reshape(-1, self.nk * 2)

            box_target = y_box_targets[i].reshape(-1, 4)
            labels = y_labels[i].reshape(-1)
            label_weights = y_label_weights[i].reshape(-1)

            kpt_targets = y_kpt_targets[i].reshape((-1, self.nk * 2))
            kpt_weights = y_kpt_weights[i].reshape((-1, self.nk * 2))

            pos_indices = ((labels >= 0) & (labels < self.nc)).nonzero(as_tuple=False).squeeze(1)
            score = label_weights.new_zeros(labels.shape)

            if len(pos_indices) > 0:
                pos_box_targets = box_target[pos_indices]
                pos_anchors = anchor[pos_indices]
                anchors_cx = (pos_anchors[:, 2] + pos_anchors[:, 0]) / 2
                anchors_cy = (pos_anchors[:, 3] + pos_anchors[:, 1]) / 2
                pos_anchor_centers = torch.stack([anchors_cx, anchors_cy], dim=-1) / stride[0]

                weight_targets = cls.detach().sigmoid()
                weight_targets = weight_targets.max(dim=1)[0][pos_indices]
                pos_decode_box_targets = pos_box_targets / stride[0]

                pos_kpt_targets = kpt_targets[pos_indices]
                pos_kpt_weights = kpt_weights.max(dim=1)[0][pos_indices] * weight_targets
                pos_kpt_weights = pos_kpt_weights.reshape((-1, 1))

                decoded_box = distance2box(pos_anchor_centers, box[pos_indices])
                decoded_kpt = self.kpt2distance(pos_anchor_centers, pos_kpt_targets / stride[0])
                score[pos_indices] = self.compute_iou(decoded_box.detach(), pos_decode_box_targets)

                # regression loss
                loss_box = self.loss_box(decoded_box, pos_decode_box_targets,
                                         weight=weight_targets, avg_factor=1.0)

                loss_kpt = self.loss_kpt(kpt[pos_indices], decoded_kpt,
                                         weight=pos_kpt_weights, avg_factor=1.0)
            else:
                loss_box = box.sum() * 0
                loss_kpt = kpt.sum() * 0
                weight_targets = torch.tensor(0).cuda()

            loss_cls = self.loss_cls(cls, (labels, score),
                                     weight=label_weights, avg_factor=num_total_samples)
            losses_cls.append(loss_cls)
            losses_box.append(loss_box)
            losses_kpt.append(loss_kpt)
            avg_factor.append(weight_targets.sum())

        avg_factor = self.reduce_mean(sum(avg_factor)).item()
        losses_box = list(map(lambda x: x / avg_factor, losses_box))
        losses_kpt = list(map(lambda x: x / avg_factor, losses_kpt))
        losses_cls = sum(i.mean() for i in losses_cls)
        losses_box = sum(i.mean() for i in losses_box)
        losses_kpt = sum(i.mean() for i in losses_kpt)

        return losses_cls, losses_box, losses_kpt

    def get_targets(self, anchors, valid_flags, y_box, y_kpt, y_cls, n):
        assert len(anchors) == len(valid_flags) == n
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchors[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(n):
            assert len(anchors[i]) == len(valid_flags[i])
            anchors[i] = torch.cat(anchors[i])
            valid_flags[i] = torch.cat(valid_flags[i])

        # compute targets for each image
        if y_cls is None:
            y_cls = [None for _ in range(n)]
        if y_kpt is None:
            y_kpt = [None for _ in range(n)]

        all_anchors = []
        all_labels = []
        all_label_weights = []
        all_box_targets = []
        all_box_weights = []
        all_kpt_targets = []
        all_kpt_weights = []
        pos_indices_list = []
        neg_indices_list = []
        for anchor, valid_flag, box, cls, kpt in zip(anchors, valid_flags, y_box, y_cls, y_kpt):
            if not valid_flag.any():
                return (None,) * 7
            # assign gt and sample anchors
            valid_anchor = anchor[valid_flag, :]

            num_level_anchors_inside = [int(i.sum()) for i in torch.split(valid_flag, num_level_anchors)]
            assign_result = self.assigner(valid_anchor, num_level_anchors_inside, box)

            pos_indices = torch.nonzero(assign_result > 0, as_tuple=False).squeeze(-1).unique()
            neg_indices = torch.nonzero(assign_result == 0, as_tuple=False).squeeze(-1).unique()

            pos_assigned_gt_indices = assign_result[pos_indices] - 1
            if box.numel() == 0:
                # hack for index error case
                assert pos_assigned_gt_indices.numel() == 0
                pos_gt_bboxes = torch.empty_like(box).view(-1, 4)
            else:
                if len(box.shape) < 2:
                    box = box.view(-1, 4)

                pos_gt_bboxes = box[pos_assigned_gt_indices, :]

            num_valid_anchors = valid_anchor.shape[0]
            box_targets = torch.zeros_like(valid_anchor, dtype=pos_gt_bboxes.dtype)
            box_weights = torch.zeros_like(valid_anchor, dtype=pos_gt_bboxes.dtype)
            kpt_targets = valid_anchor.new_zeros(size=(valid_anchor.shape[0], self.nk * 2), dtype=pos_gt_bboxes.dtype)
            kpt_weights = valid_anchor.new_zeros(size=(valid_anchor.shape[0], self.nk * 2), dtype=pos_gt_bboxes.dtype)
            labels = valid_anchor.new_full((num_valid_anchors,), self.nc, dtype=torch.long)
            label_weights = valid_anchor.new_zeros(num_valid_anchors, dtype=torch.float)

            if len(pos_indices) > 0:
                box_targets[pos_indices, :] = pos_gt_bboxes
                box_weights[pos_indices, :] = 1.0

                kpt_targets[pos_indices, :] = kpt[pos_assigned_gt_indices, :, :2].reshape((-1, self.nk * 2))
                kpt_weights[pos_indices, :] = kpt[pos_assigned_gt_indices, :, 2].mean(dim=1, keepdims=True)

                if cls is None:
                    # Only rpn gives cls as None Foreground is the first class
                    labels[pos_indices] = 0
                else:
                    labels[pos_indices] = cls[pos_assigned_gt_indices]
                label_weights[pos_indices] = 1.0

            if len(neg_indices) > 0:
                label_weights[neg_indices] = 1.0

            # map up to original set of anchors
            n = anchor.size(0)
            valid_anchor = self.unmap(valid_anchor, n, valid_flag)
            labels = self.unmap(labels, n, valid_flag, self.nc)
            box_targets = self.unmap(box_targets, n, valid_flag)
            box_weights = self.unmap(box_weights, n, valid_flag)
            label_weights = self.unmap(label_weights, n, valid_flag)

            kpt_targets = self.unmap(kpt_targets, n, valid_flag)
            kpt_weights = self.unmap(kpt_weights, n, valid_flag)
            all_anchors.append(valid_anchor)
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_box_targets.append(box_targets)
            all_box_weights.append(box_weights)
            all_kpt_targets.append(kpt_targets)
            all_kpt_weights.append(kpt_weights)
            pos_indices_list.append(pos_indices)
            neg_indices_list.append(neg_indices)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(i.numel(), 1) for i in pos_indices_list])
        num_total_neg = sum([max(i.numel(), 1) for i in neg_indices_list])
        # split targets to a list w.r.t. multiple levels
        all_anchors = self.to_levels(all_anchors, num_level_anchors)
        all_labels = self.to_levels(all_labels, num_level_anchors)
        all_label_weights = self.to_levels(all_label_weights, num_level_anchors)
        all_box_targets = self.to_levels(all_box_targets, num_level_anchors)
        all_box_weights = self.to_levels(all_box_weights, num_level_anchors)
        all_kpt_targets = self.to_levels(all_kpt_targets, num_level_anchors)
        all_kpt_weights = self.to_levels(all_kpt_weights, num_level_anchors)
        return (all_anchors,
                all_labels, all_label_weights,
                all_box_targets, all_box_weights,
                all_kpt_targets, all_kpt_weights, num_total_pos, num_total_neg)

    @staticmethod
    def kpt2distance(points, kps, max_dis=None, eps=0.1):
        output = []
        for i in range(0, kps.shape[1], 2):
            px = kps[:, i] - points[:, i % 2]
            py = kps[:, i + 1] - points[:, i % 2 + 1]
            if max_dis is not None:
                px = px.clamp(min=0, max=max_dis - eps)
                py = py.clamp(min=0, max=max_dis - eps)
            output.append(px)
            output.append(py)
        return torch.stack(output, dim=-1)

    @staticmethod
    def compute_iou(boxes1, boxes2, eps=1e-6):
        assert (boxes1.size(-1) == 4 or boxes1.size(0) == 0)
        assert (boxes2.size(-1) == 4 or boxes2.size(0) == 0)

        assert boxes1.shape[:-2] == boxes2.shape[:-2]
        batch_shape = boxes1.shape[:-2]

        rows = boxes1.size(-2)
        cols = boxes2.size(-2)
        assert rows == cols

        if rows * cols == 0:
            return boxes1.new(batch_shape + (rows,))

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        union = area1 + area2 - overlap

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        return overlap / union

    @staticmethod
    def to_levels(target, num_levels):
        target = torch.stack(target, 0)
        targets = []
        i = 0
        for n in num_levels:
            j = i + n
            targets.append(target[:, i:j])
            i = j
        return targets

    @staticmethod
    def reduce_mean(x):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return x
        x = x.clone()
        torch.distributed.all_reduce(x.div_(torch.distributed.get_world_size()),
                                     op=torch.distributed.ReduceOp.SUM)
        return x

    @staticmethod
    def unmap(data, count, indices, fill=0):
        if data.dim() == 1:
            output = data.new_full((count,), fill)
            output[indices.type(torch.bool)] = data
        else:
            size = (count,) + data.size()[1:]
            output = data.new_full(size, fill)
            output[indices.type(torch.bool), :] = data
        return output


class NMS:
    def __init__(self, params, conf_threshold=0.02):
        self.nc = 1
        self.nk = 5
        self.anchor_generator = AnchorGenerator(params['face_anchors']['strides'],
                                                params['face_anchors']['ratios'],
                                                params['face_anchors']['scales'],
                                                params['face_anchors']['sizes'])
        self.conf_threshold = conf_threshold

    def __call__(self, outputs):
        x_cls, x_box, x_kpt = outputs
        assert len(x_cls) == len(x_box)
        num_levels = len(x_cls)
        n = x_cls[0].shape[0]
        device = x_cls[0].device
        sizes = [x_cls[i].shape[-2:] for i in range(num_levels)]
        anchors = self.anchor_generator.make_anchors(sizes, device=device)

        results = []
        for j in range(n):
            x_cls_list = [x_cls[i][j].detach() for i in range(num_levels)]
            x_box_list = [x_box[i][j].detach() for i in range(num_levels)]
            x_kpt_list = [x_kpt[i][j].detach() for i in range(num_levels)]

            y_kpt = []
            y_box = []
            y_cls = []
            assert len(x_cls) == len(x_box) == len(anchors)

            for cls, box, kpt, stride, anchor in zip(x_cls_list, x_box_list, x_kpt_list,
                                                     self.anchor_generator.strides, anchors):
                assert cls.size()[-2:] == box.size()[-2:]
                assert stride[0] == stride[1]

                box = box.permute(1, 2, 0)
                kpt = kpt.permute(1, 2, 0)
                box = box.reshape((-1, 4)) * stride[0]
                kpt = kpt.reshape((-1, 2 * self.nk)) * stride[0]

                cx = (anchor[:, 2] + anchor[:, 0]) / 2
                cy = (anchor[:, 3] + anchor[:, 1]) / 2
                anchor = torch.stack(tensors=[cx, cy], dim=-1)

                cls = cls.permute(1, 2, 0).reshape(-1, self.nc).sigmoid()
                box = distance2box(anchor, box)
                kpt = distance2kpt(anchor, kpt)

                y_cls.append(cls)
                y_box.append(box)
                y_kpt.append(kpt)

            y_cls = torch.cat(y_cls)
            y_box = torch.cat(y_box)
            y_kpt = torch.cat(y_kpt)
            y_cls = torch.cat(tensors=[y_cls, y_cls.new_zeros(y_cls.shape[0], 1)], dim=1)

            results.append(self.__nms(y_cls, y_box, y_kpt))
        return results

    def __nms(self, cls, box, kpt):
        scores, indices = torch.max(cls, 1)
        valid_mask = scores >= self.conf_threshold

        box = box[valid_mask]
        kpt = kpt[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]

        if indices.numel() == 0:
            return torch.cat(tensors=(box, indices), dim=-1), kpt
        else:
            box, keep = self.__batched_nms(box, scores, indices)
            return torch.cat(tensors=(box, indices[keep][:, None]), dim=-1), kpt[keep]

    @staticmethod
    def __batched_nms(boxes, scores, indices):
        max_coordinate = boxes.max()
        offsets = indices.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        if len(boxes_for_nms) < 10_000:
            keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold=0.45)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            for i in torch.unique(indices):
                mask = (indices == i).nonzero(as_tuple=False).view(-1)
                keep = torchvision.ops.nms(boxes_for_nms[mask], scores[mask], iou_threshold=0.45)
                total_mask[mask[keep]] = True

            keep = total_mask.nonzero(as_tuple=False).view(-1)
            keep = keep[scores[keep].argsort(descending=True)]
            boxes = boxes[keep]
            scores = scores[keep]

        return torch.cat(tensors=[boxes, scores[:, None]], dim=-1), keep
