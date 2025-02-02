import os
import random

import cv2
import numpy
import torch
from PIL import Image

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, r, (w, h)


class Resize:
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image, box, kpt):
        image, scale = self.resize_image(image, self.input_size)
        box = self.resize_box(box, image.shape, scale)
        kpt = self.resize_kpt(kpt, image.shape, scale)
        return image, box, kpt

    def resize_image(self, image, size):
        h, w = image.shape[:2]

        image = cv2.resize(image, dsize=(size, size), interpolation=self.resample())
        scale = numpy.array(object=[size / w, size / h, size / w, size / h], dtype=numpy.float32)
        return image, scale

    @staticmethod
    def resize_box(box, shape, scale_factor):
        box = box * scale_factor
        box[:, 0::2] = numpy.clip(box[:, 0::2], 0, shape[1])
        box[:, 1::2] = numpy.clip(box[:, 1::2], 0, shape[0])
        return box

    @staticmethod
    def resize_kpt(kpt, shape, factors):
        assert factors[0] == factors[2]
        assert factors[1] == factors[3]
        kpt[:, :, 0] *= factors[0]
        kpt[:, :, 1] *= factors[1]
        kpt[:, :, 0] = numpy.clip(kpt[:, :, 0], 0, shape[1])
        kpt[:, :, 1] = numpy.clip(kpt[:, :, 1], 0, shape[0])
        return kpt

    @staticmethod
    def resample():
        choices = (cv2.INTER_AREA,
                   cv2.INTER_CUBIC,
                   cv2.INTER_LINEAR,
                   cv2.INTER_NEAREST,
                   cv2.INTER_LANCZOS4)
        return random.choice(seq=choices)


class RandomHSV:
    def __init__(self, params):
        self.h = params['hsv_h']
        self.s = params['hsv_s']
        self.v = params['hsv_v']

    def __call__(self, image):
        r = numpy.random.uniform(low=-1, high=1, size=3) * [self.h, self.s, self.v] + 1
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        x = numpy.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype('uint8')
        lut_s = numpy.clip(x * r[1], a_min=0, a_max=255).astype('uint8')
        lut_v = numpy.clip(x * r[2], a_min=0, a_max=255).astype('uint8')

        hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


class RandomCrop:
    def __init__(self):
        self.crop_choice = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

    @staticmethod
    def is_center_of_bboxes_in_patch(boxes, x1, y1, x2, y2):
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        return ((center[:, 0] > x1) *
                (center[:, 1] > y1) *
                (center[:, 0] < x2) *
                (center[:, 1] < y2))

    def __call__(self, image, box, cls, kpt):

        h, w, c = image.shape
        scale_retry = 0

        max_scale = numpy.amax(self.crop_choice)

        while True:
            scale_retry += 1

            if scale_retry == 1 or max_scale > 1.0:
                scale = numpy.random.choice(self.crop_choice)
            else:
                scale = scale * 1.2

            for i in range(250):
                short_side = min(w, h)
                c_w = int(scale * short_side)
                c_h = c_w

                if w == c_w:
                    x1 = 0
                elif w > c_w:
                    x1 = numpy.random.randint(0, w - c_w)
                else:
                    x1 = numpy.random.randint(w - c_w, 0)

                if h == c_h:
                    y1 = 0
                elif h > c_h:
                    y1 = numpy.random.randint(0, h - c_h)
                else:
                    y1 = numpy.random.randint(h - c_h, 0)

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x1 + c_w)
                y2 = int(y1 + c_h)

                mask = self.is_center_of_bboxes_in_patch(box, x1, y1, x2, y2)
                if not mask.any():
                    continue

                mask = self.is_center_of_bboxes_in_patch(box, x1, y1, x2, y2)
                box = box[mask]
                box[:, 2:] = box[:, 2:].clip(max=(x2, y2))
                box[:, :2] = box[:, :2].clip(min=(x1, y1))
                box -= numpy.tile((x1, y1), 2)

                cls = cls[mask]

                kpt = kpt.copy()
                kpt = kpt[mask, :, :]
                kpt[:, :, :2] = kpt[:, :, :2].clip(max=(x2, y2))
                kpt[:, :, :2] = kpt[:, :, :2].clip(min=(x1, y1))
                kpt[:, :, 0] -= x1
                kpt[:, :, 1] -= y1

                image_ones = numpy.ones(shape=(c_h, c_w, 3), dtype=image.dtype) * 128
                a = [x1, y1, x2, y2]
                a[0] = max(0, x1)
                a[1] = max(0, y1)
                a[2] = min(image.shape[1], x2)
                a[3] = min(image.shape[0], y2)
                b = [x1, y1, x2, y2]
                b[0] = max(0, -1 * x1)
                b[1] = max(0, -1 * y1)
                b[2] = b[0] + (a[2] - a[0])
                b[3] = b[1] + (a[3] - a[1])
                image_ones[b[1]:b[3], b[0]:b[2], :] = image[a[1]:a[3], a[0]:a[2], :]

                return image_ones, box, cls, kpt


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations
            self.transform = albumentations.OneOf([albumentations.ToGray(),
                                                   albumentations.Emboss(),
                                                   albumentations.Sharpen(),
                                                   albumentations.ToSepia(),
                                                   albumentations.ISONoise(),
                                                   albumentations.RGBShift(),
                                                   albumentations.FancyPCA(),
                                                   albumentations.ImageCompression(quality_lower=50,
                                                                                   quality_upper=99)])

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image):
        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']
        return image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.nk = 5
        self.params = params
        self.augment = augment
        self.input_size = input_size
        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)

        self.random_crop = RandomCrop()
        self.random_hsv = RandomHSV(self.params)
        self.resize = Resize(self.input_size)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        label = self.labels[index].copy()
        image = cv2.imread(self.filenames[index])

        cls = label[:, 0:1].reshape(-1)
        box = label[:, 1:5]
        kpt = label[:, 5:].reshape(-1, self.nk, 3)

        if self.augment:
            self.random_hsv(image)
            image, box, cls, kpt = self.random_crop(image, box, cls, kpt)
            image, box, kpt = self.resize(image, box, kpt)

            # Flip left-right
            if random.random() < 0.5:
                image, box, kpt = self.flip_lr(image, box, kpt, image.shape)
            # Albumentations, COLOR augmentations only
            image = self.albumentations(image)
        else:
            image, ratio, pad = resize(image, self.input_size)
            box[:, [0, 2]] = box[:, [0, 2]] * ratio + pad[0]
            box[:, [1, 3]] = box[:, [1, 3]] * ratio + pad[1]
            kpt[:, :, 0] = kpt[:, :, 0] * ratio + pad[0]
            kpt[:, :, 1] = kpt[:, :, 1] * ratio + pad[1]

        nl = len(box)
        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        target_kpt = torch.zeros((nl, self.nk, 3))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)
            target_kpt = torch.from_numpy(kpt)
        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample, dtype='float32')
        return torch.from_numpy(sample), target_cls, target_box, target_kpt, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def flip_lr(image, box, kpt, shape):
        w = shape[1]
        image = numpy.flip(image, axis=1)
        f_box = box.copy()
        f_box[..., 0::4] = w - box[..., 2::4]
        f_box[..., 2::4] = w - box[..., 0::4]

        f_kpt = kpt.copy()
        flip_order = [1, 0, 2, 4, 3]
        for i, a in enumerate(flip_order):
            f_kpt[:, i, :] = kpt[:, a, :]
        f_kpt[..., 0] = w - f_kpt[..., 0]
        return image, f_box, f_kpt

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                size = image.size  # image size
                assert (size[0] > 32) & (size[1] > 32), f'image size {size} < 32 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        continue
                else:
                    continue
            except FileNotFoundError:
                continue
            except AssertionError:
                print('AssertionError')
                continue
            x[filename] = label
        torch.save(x, path)
        return x

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, kpt, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)
        kpt = torch.cat(kpt, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'kpt': kpt,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets
