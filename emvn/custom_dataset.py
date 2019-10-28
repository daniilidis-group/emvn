import os
import time
import numpy as np
import re
from pathlib import Path

import cv2
import tensorpack.dataflow as df

import constants as cts


def find_classes(dir, filter_classes=None):
    classes = [d for d in os.listdir(dir)
               if os.path.isdir(os.path.join(dir, d))]
    if filter_classes:
        classes = [c for c in classes if c in filter_classes]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def autocrop(im, keep_aspect_ratio=False):
    cols = np.where(im.max(axis=0) > 0)[0]
    rows = np.where(im.max(axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        print('WARNING! zero img detected!')
        return im
    else:
        crop = (min(rows), max(rows),
                min(cols), max(cols))
        if keep_aspect_ratio:
            y0, y1, x0, x1 = crop
            h, w = y1 - y0, x1 - x0
            if h > w:
                d = (h - w) / 2
                x0 -= int(np.floor(d))
                x1 += int(np.ceil(d))
            elif h < w:
                d = (w - h) / 2
                y0 -= int(np.floor(d))
                y1 += int(np.ceil(d))
            h, w = im.shape[:2]
            crop = (max(0, y0), min(h - 1, y1),
                    max(0, x0), min(w - 1, x1))

        return im[crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]


class LMDBMultiView():
    def _augment(x):
        # brighness / contrast / noise
        factors = np.random.uniform(0.6, 1.4, 2)
        # contrast
        mean = x.mean()
        x = (x - mean) * factors[0] + mean
        # brightness
        x = x * factors[1]
        # Gaussian noise sigma=0.1 (seems to hurt val performance)
        # x += np.random.randn(*x.shape) * 0.1 * 255
        # TODO: add jpeg compression (40-100)?
        # TODO: add small affine transforms?

        x = x.clip(0, 255).astype('uint8')

        return x

    def load(self, xs):
        if self.filter_classes:
            if not any([c in xs[-1] for c in self.filter_classes]):
                # print('removing ' + xs[-1])
                return None

        fname = xs[-1]

        if self.filter_ids or self.filter_classes:
            match = re.match('(.*)_(\d\d\d\d).*', Path(fname).name)
            if match:
                cls, id = match.groups()
            else:
                print('Warning! could not match class and id!')
                cls, id = self.filter_classes[0], 0

        if self.filter_ids:
            if int(id) > self.filter_ids[cls]:
                return None

        label = xs[-2]
        if self.label_to0idx:
            label -= 1
        # adjust label
        if self.filter_classes:
            label = self.filter_classes.index(cls)

        # this is when we have n encoded views
        # xs = np.stack([cv2.imdecode(x, cv2.IMREAD_GRAYSCALE)
        #                for x in xs[:-2]])
        # and this when we have a single encoding for all views (faster)
        flag = cv2.IMREAD_COLOR if self.rgb else cv2.IMREAD_GRAYSCALE
        x = cv2.imdecode(xs[0], flag)
        xs = np.split(x, x.shape[1] // x.shape[0], axis=1)

        if self.filter_views is not None:
            if len(xs) == len(self.filter_views):
                # upright datasets correspond to views given by this constant
                xs = [xs[i] for i in cts.upright_to_homogeneous[len(xs)]]
            else:
                xs = [xs[i] for i in self.filter_views]

        if self.autocrop:
            xs = [autocrop(x, self.keep_aspect_ratio) for x in xs]

        if self.force_res > 0:
            out = []
            res = self.force_res
            for x in xs:
                h, w = x.shape
                if (h != res or w != res):
                    x = cv2.resize(x, (res, res))
                out.append(x)
            xs = out

        # if we want to augment individual views, add here
        if self.polarmode == 'logpolar':
            w = xs[0].shape[0]
            # this will create some dead pixels
            m = w / np.log(w * np.sqrt(2) / 2)
            # but this may crop parts of the original img:
            # m = w/np.log(w/2)
            xs = [cv2.logPolar(x, ((w - 1.) / 2., (w - 1.) / 2.), m,
                               cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
                  for x in xs]
        elif self.polarmode == 'polar':
            w = xs[0].shape[0]
            m = w * np.sqrt(2) / 2
            xs = [cv2.linearPolar(x, ((w - 1.) / 2., (w - 1.) / 2.), m,
                                  cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
                  for x in xs]

        xs = np.stack(xs, axis=0)

        return [xs, label, fname]

    def __init__(self, datafile, batch_size,
                 num_workers=1, nviews=12, reset=True, augment=False,
                 filter_classes=None, filter_views=None, polarmode='cartesian',
                 shuffle=True, filter_ids=None, label_to0idx=False,
                 rgb=False, force_res=0, autocrop=False, keep_aspect_ratio=False):
        self.filter_classes = filter_classes
        self.filter_views = filter_views
        self.filter_ids = filter_ids
        self.polarmode = polarmode
        self.label_to0idx = label_to0idx
        self.rgb = rgb
        self.force_res = force_res
        self.autocrop = autocrop
        self.keep_aspect_ratio = keep_aspect_ratio

        if not isinstance(datafile, list):
            datafile = [datafile]

        ds = []
        for d in datafile:
            ds.append(df.LMDBSerializer.load(d, shuffle=shuffle))
            if shuffle:
                ds[-1] = df.LocallyShuffleData(ds[-1], 100)
            ds[-1] = df.PrefetchData(ds[-1], 20, 1)

            ds[-1] = df.MapData(ds[-1], self.load)
            if augment:
                ds[-1] = df.MapDataComponent(ds[-1], LMDBMultiView._augment, 0)

            if (not filter_classes and
                not filter_ids and
                    num_workers > 1):
                # warning: skipping this is slower when filtering datasets
                #          but epoch counting will be wrong otherwise
                ds[-1] = df.PrefetchDataZMQ(ds[-1], num_workers)
            ds[-1] = df.BatchData(ds[-1], batch_size)

            if reset:
                ds[-1].reset_state()

        self.ds = ds

    def __iter__(self):
        t0 = time.perf_counter()
        for d in self.ds:
            for e in d:
                t = time.perf_counter() - t0
                t0 = time.perf_counter()
                yield [*e, t]

    def __len__(self):
        return sum([len(d) for d in self.ds])


def load_data(args):
    if args.data.endswith('.lmdb'):
        return load_data_lmdb(args)
    elif args.data == 'fake':
        nviews = args.n_homogeneous if args.n_homogeneous else args.n_group_elements
        imshape = (args.batch_size, nviews, 224, 224)
        train_loader = df.FakeData((imshape, [args.batch_size], [1], [1]),
                                   dtype=('uint8', 'int64', 'str', 'float32'))
        test_loader = df.FakeData((imshape, [args.batch_size], [1], [1]),
                                  dtype=('uint8', 'int64', 'str', 'float32'))
        train_loader.reset_state()
        test_loader.reset_state()
        return (train_loader, test_loader, np.arange(40))


def get_lmdb_properties(args):
    ds = df.LMDBSerializer.load(args.data.format('test'),
                                shuffle=False)
    ds.reset_state()
    labels = []
    for d in ds:
        labels.append(d[-2])
    # component -2 is label, -1 is fname
    classes = np.unique(labels)
    nviews = len(d) - 2

    return classes, nviews


def get_filtered_views(args):
    if args.n_homogeneous > 0:
        filter_views = (cts.homogeneous_tables
                        [args.n_group_elements]
                        [args.n_homogeneous]
                        ['ids'])
    else:
        filter_views = None

    return filter_views


def load_data_lmdb(args):
    classes, nviews = get_lmdb_properties(args)
    if classes.min() == 1:
        print('No 0 class detected! Subtracting 1 from labels!')
        classes -= 1
        label_to0idx = True
        assert not args.filter_classes
    else:
        label_to0idx = False

    if args.filter_classes:
        classes = np.arange(len(args.filter_classes))
    print('Classes={:d}, Views={:d}'.format(len(classes), nviews))

    filter_views = get_filtered_views(args)
    filter_ids = {}
    for m in ['train', 'test']:
        filter_ids[m] = cts.modelnet_subset[m] if args.modelnet_subset else None

    traindata = ([args.data.format('train'), args.data.format('val')]
                 if args.combine_train_val
                 else args.data.format('train'))
    train_loader = LMDBMultiView(traindata,
                                 args.batch_size,
                                 num_workers=8,
                                 nviews=nviews,
                                 augment=args.augment,
                                 filter_classes=args.filter_classes,
                                 filter_views=filter_views,
                                 polarmode=args.polarmode,
                                 shuffle=True,
                                 filter_ids=filter_ids['train'],
                                 label_to0idx=label_to0idx,
                                 rgb=args.rgb,
                                 force_res=args.force_res,
                                 autocrop=args.autocrop,
                                 keep_aspect_ratio=args.keep_aspect_ratio)
    test_loader = LMDBMultiView(args.data.format('test'),
                                # WARNING! we halve the batch size so eval
                                # can run on Titan Z
                                args.batch_size // 2,
                                num_workers=1,
                                nviews=nviews,
                                augment=False,
                                filter_classes=args.filter_classes,
                                filter_views=filter_views,
                                polarmode=args.polarmode,
                                shuffle=False,
                                filter_ids=filter_ids['test'],
                                label_to0idx=label_to0idx,
                                rgb=args.rgb,
                                force_res=args.force_res,
                                autocrop=args.autocrop,
                                keep_aspect_ratio=args.keep_aspect_ratio)

    return train_loader, test_loader, classes
