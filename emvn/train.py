import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform

import argparse
import numpy as np
import time
import os
import shutil
import subprocess
from pathlib import Path

import models
import constants as cts

import util
from logger import Logger
import custom_dataset

start_time = time.perf_counter()

MODELS = ['svcnn', 'mvcnn', 'resnetmvcnn', 'resnetmvgcnn', 'resnetsvcnn']
ACTIVATIONS = {'relu': nn.ReLU,
               'prelu': nn.PReLU,
               'leakyrelu': nn.LeakyReLU}

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18], type=int,
                    metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=60, type=int,
                    metavar='N', help='mini-batch size (default: 60)')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'nesterov'],
                    help='Optimizer to use')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.00005)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')  # ?
parser.add_argument('--lr-decay-mode', default='step', type=str, choices=['step', 'cos'],
                    help='learning rate decay mode')
parser.add_argument('--lr-decay-freq', default=10, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--weight_decay', default=0., type=float,
                    metavar='W', help='weight decay (regularization)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--warm_start_from', '--ws', default='', type=str, metavar='PATH',
                    help='Warm-start from checkpoint')
parser.add_argument('--pretrained', dest='pretrained',
                    action='store_true', help='use pre-trained model')
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.add_argument('--logdir', default='logs', type=str, metavar='PATH')
parser.add_argument('-c', '--check_file',
                    default='checkpointmv', type=str, metavar='PATH')
parser.add_argument('--latest_fname', default='latest.pth.tar', type=str,
                    help='Latest checkpoint fname')

parser.add_argument('--viewpool', '--vp', default='max', type=str,
                    choices=['max', 'avg',
                             'verylate_avg', 'verylate_max',
                             'mid_max', 'mid_avg'])
parser.add_argument('--exit_after', '--ea', default=np.inf, type=float,
                    help='do not start another epoch if total running time exceed limit.')
parser.add_argument('--gconv_channels', '--gcc', default=[],
                    type=lambda x: [int(i) for i in x.split(',')],
                    help='Number of channels per group conv layer.')
parser.add_argument('--gconv_support', default=None,
                    type=lambda x: [int(i) for i in x.split(',')],
                    help='Localized filter support (nonzero indices).')
parser.add_argument('--freeze_layers', default=[],
                    type=lambda x: x.split(','),
                    help='Layers to freeze (no gradients)')
parser.add_argument('--skip_train', dest='skip_train', action='store_true',
                    help='skip training step')
parser.add_argument('--skip_eval', dest='skip_eval', action='store_true',
                    help='skip eval step')
parser.add_argument('--noskip_eval', dest='skip_eval', action='store_false',
                    help='do not skip eval step')
parser.add_argument('--view_dropout', default=[],
                    type=lambda x: [int(i) for i in x.split(',')],
                    help='Min/max views to dropout per batch')
parser.add_argument('--n_group_elements', default=12, type=int, choices=[12, 60],
                    help='Number of elements in the group for group conv')
parser.add_argument('--max_steps', default=0, type=int,
                    help='Stop after # steps per epoch')
parser.add_argument('--bn_after_gconv', default=True)
parser.add_argument('--nobn_after_gconv', dest='bn_after_gconv', action='store_false',
                    help='Apply batch norm after group convs')
parser.add_argument('--gconv_activation', type=str,
                    choices=['relu', 'leakyrelu', 'prelu'], default='relu',
                    help='Apply batch norm after group convs')
parser.add_argument('--n_fc_before_gconv', type=int, default=0,
                    help='Apply MLP before GCONV')
parser.add_argument('--filter_classes', default=[],
                    type=lambda x: [str(i) for i in x.split(',')],
                    help='Return only selected classes')
parser.add_argument('--n_classes', type=int, default=0,
                    help='number of classes; try to autodetect if zero')
parser.add_argument('--rgb', dest='rgb', action='store_true',
                    help='Indication of RGB dataset')
parser.add_argument('--force_res', type=int, default=0,
                    help='Resize input to given resolution')
parser.add_argument('--autocrop', action='store_true',
                    help='Crop inputs usig tight bounding boxes.')
parser.add_argument('--keep_aspect_ratio', action='store_true',
                    help='Keep aspect ratio when cropping.')
parser.add_argument('--modelnet_subset', dest='modelnet_subset', action='store_true',
                    help='Use smaller ModelNet subset')
parser.add_argument('--combine_train_val', dest='combine_train_val', action='store_true',
                    help='Combine train and validation sets')

parser.add_argument('--n_homogeneous', '--nhs', type=int, default=0,
                    help='If > 0, consider homogeneous space of nhs elements instead of group.')
parser.add_argument('--label_smoothing', type=float, default=0,
                    help='Label smoothing level')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='Do augmentation')
parser.add_argument('--logpolar', dest='logpolar', action='store_true',
                    help='Convert views to logpolar coordinates')
parser.add_argument('--polar', dest='polar', action='store_true',
                    help='Convert views to polar coordinates')
parser.add_argument('--circpad', dest='circpad', action='store_true',
                    help='Use circular padding (useful for logpolar representations)')
parser.add_argument('--eval_retrieval', dest='eval_retrieval', action='store_true',
                    default=True, help='Also Evaluate retrieval')
parser.add_argument('--noeval_retrieval',
                    dest='eval_retrieval', action='store_false')
parser.add_argument('--retrieval_include_same', dest='retrieval_include_same',
                    action='store_true', default=False,
                    help='Always include elements of same class first when evaluating retrieval.')
parser.add_argument('--retrieval_dir', default='',
                    help='Name of retrieval directory (eg test_normal and so on for SHREC17)')
parser.add_argument('--save_fmaps', dest='save_fmaps', action='store_true',
                    help='save feature maps')
parser.add_argument('--nolog', dest='nolog', action='store_true',
                    help='do not save logs')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='run on cpu only')
parser.add_argument('--homogeneous_only1st', dest='homogeneous_only1st', action='store_true',
                    help='Run homogeneous conv on 1st G-CNN layer only')
# TODO: this should be a weight, not T/F
parser.add_argument('--triplet_loss', dest='triplet_loss', action='store_true',
                    help='Use triplet loss')


args = parser.parse_args()
# TODO: make check_args()
if args.augment:
    assert args.data.endswith('.lmdb')

args.polarmode = 'cartesian'
if args.logpolar:
    assert args.data.endswith('.lmdb')
    assert not args.polar
    args.polarmode = 'logpolar'
if args.polar:
    assert args.data.endswith('.lmdb')
    assert not args.logpolar
    args.polarmode = 'polar'

if args.modelnet_subset:
    assert args.data.endswith('.lmdb')
if args.combine_train_val:
    assert args.data.endswith('.lmdb')
if args.rgb:
    assert args.data.endswith('.lmdb')
if args.force_res:
    assert args.data.endswith('.lmdb')


logname = 'logging.log'
if args.skip_train:
    logname = 'logging_eval.log'
if args.nolog:
    logname = ''
logger = util.init_logger(args.logdir, fname=logname)


def load_model(classes):
    model = models.resnet_mvgcnn(args.depth,
                                 pretrained=args.pretrained,
                                 num_classes=len(classes),
                                 gconv_channels=args.gconv_channels,
                                 view_dropout=args.view_dropout,
                                 n_group_elements=args.n_group_elements,
                                 n_homogeneous=args.n_homogeneous,
                                 bn_after_gconv=args.bn_after_gconv,
                                 gconv_activation=ACTIVATIONS[args.gconv_activation],
                                 gconv_support=args.gconv_support,
                                 viewpool=args.viewpool,
                                 n_fc_before_gconv=args.n_fc_before_gconv,
                                 circpad=args.circpad,
                                 full_homogeneous=not args.homogeneous_only1st)

    logger.info('Loaded model...')

    return model


# Helper functions
def load_checkpoint(strdir, model, optimizer):
    if not os.path.isfile(strdir):
        logger.info('WARNING: no checkpoint file found! starting from scratch')
        return

    checkpoint = torch.load(strdir)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError:
        logger.info('Warning! could not load optimizer state; this may happen when warm_starting from different model')

    train_state = dict(start_epoch=checkpoint['epoch'],
                       lr=checkpoint['lr'],
                       step=checkpoint['step'],
                       steps_epoch=checkpoint.get('steps_epoch', 0))

    return train_state


def np2tensor(x, device):
    try:
        x = torch.from_numpy(x)
    except TypeError:
        # already a Tensor
        pass
    if not args.cpu:
        x.pin_memory()
        return x.cuda(device, non_blocking=True)
    else:
        return x


# TODO: what's a better place to keep these?
m40_mean = torch.from_numpy(np.array(cts.m40_mean, dtype='float32'))
m40_std = torch.from_numpy(np.array(cts.m40_std, dtype='float32'))


def load_inputs_targets(inputs, targets, device):
    # CHECKME! not sure if this is the most efficient way
    inputs, targets = [np2tensor(i, device) for i in [inputs, targets]]
    # dims: batch, views, channels, rows, cols
    # gray to RGB (TODO: this should be weighted)
    # lmdb datasets are 1 channel
    if args.data.endswith('.lmdb'):  # and inputs.shape[-3] != 3:
        if inputs.shape[-1] == 3:
            inputs = inputs.transpose(-1, -2).transpose(-2, -3)
        else:
            inputs = torch.stack([inputs, inputs, inputs], dim=-3)
    else:
        inputs = torch.transpose(torch.transpose(inputs, -1, -2), -2, -3)
    inputs = inputs.float() / 255.
    # TODO: change this for matterport!!!
    inputs -= m40_mean[:, None, None].to(device)
    inputs /= m40_std[:, None, None].to(device)
    targets = targets.long()

    return inputs, targets


def smooth_labels(targets, device):
    # TODO! take n_classes as arg!
    n_classes = 40
    y_onehot = (torch.FloatTensor(targets.shape[0], n_classes)
                .cuda(device, non_blocking=True))
    y_onehot.fill_(args.label_smoothing / (n_classes - 1.))
    onehot_targets = y_onehot.scatter_(1, targets[:, None],
                                       1 - args.label_smoothing)

    return onehot_targets


def compute_triplet_loss(embeddings, targets, samples, margin=0.2, mode='hardest', n_classes=None):
    embeddings = F.normalize(embeddings, dim=1)
    cpu_targets = targets.cpu().numpy()
    if len(samples) < n_classes:
        loss = 0
    else:
        # compute pairwise cosine distances
        sample_matrix = torch.stack([v for k, v in sorted(samples.items())])
        dists = 1 - (embeddings @ torch.transpose(sample_matrix, 1, 0))
        positive = torch.gather(dists, dim=1, index=targets[:, None])[:, 0]

        # TODO: can we do this using Tensors?
        cls = set(np.arange(n_classes))
        complement = np.stack([list(cls - {t}) for t in cpu_targets])
        negative = torch.gather(dists,
                                dim=1,
                                index=torch.from_numpy(complement).to(dists.device))
        if mode == 'hardest':
            # hardest sample: min dist from wrong class
            negative = torch.min(negative, dim=1)[0]
        elif mode == 'semi-hard':
            # semi-hard: wrong class but closer than 'positive'
            raise NotImplementedError()

        loss = torch.mean(torch.max(positive - negative + margin,
                                    torch.zeros_like(positive)))

    # update samples
    for e, t in zip(embeddings, cpu_targets):
        samples[t] = e.detach()

    return loss, samples


def train(train_loader, device, model, criterion, optimizer, summary_writer, train_state, n_classes=None):
    train_size = len(train_loader)
    t_batch = time.perf_counter()
    triplet_samples = {}
    for i, (inputs, targets, fnames, tload) in enumerate(train_loader):
        t0 = time.perf_counter()
        inputs, targets = load_inputs_targets(inputs, targets, device)
        outputs = model(inputs)
        if args.label_smoothing > 0:
            onehot_targets = smooth_labels(targets, device)
            loss = criterion(outputs, onehot_targets)
        else:
            loss = criterion(outputs, targets)

        if args.triplet_loss:
            triplet_loss, triplet_samples = compute_triplet_loss(outputs,
                                                                 targets,
                                                                 triplet_samples,
                                                                 n_classes=n_classes)
            # TODO: use weights here
            loss += triplet_loss
        else:
            triplet_loss = None

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0

        # add summaries
        pred = torch.argmax(outputs.data, 1).cpu()
        acc = 100. * (pred == targets.cpu()).sum().double() / targets.size(0)
        ims = get_summary_ims(inputs) if train_state['step'] == 0 else None
        train_state['step'] += 1

        if not args.nolog:
            util.logStep(summary_writer, model,
                         train_state['step'],
                         loss, acc, train_state['lr'], 1. / elapsed,
                         ims, triplet_loss=triplet_loss)

        # update lr if 'cosine' mode
        if args.lr_decay_mode == 'cos':
            # estimate steps_epoch from first epoch (we may have dropped entries)
            steps_epoch = (train_state['steps_epoch'] if train_state['steps_epoch'] > 0
                           else len(train_loader))
            # TODO: there will be a jump here if many entries are dropped
            #       and we only figure out # of steps after first epoch

            if train_state['step'] < steps_epoch:
                train_state['lr'] = args.lr * train_state['step'] / steps_epoch
            else:
                nsteps = steps_epoch * args.epochs
                train_state['lr'] = (0.5 * args.lr *
                                     (1 + np.cos(train_state['step'] * np.pi / nsteps)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = train_state['lr']

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f Time batch: %.4f s; Time GPU %.4f s; Time to load: %.4f s" %
                  (i + 1, train_size, loss.item(),
                   time.perf_counter() - t_batch,
                   elapsed,
                   np.array(tload).mean()))
        t_batch = time.perf_counter()

        if args.max_steps > 0 and i > args.max_steps:
            break


def save_shrec17_output(query, dists, fnames, query_lbl, lbls):
    # WARNING! not too flexible; we assume fixed dset names
    outdir = os.path.join(args.logdir, 'retrieval', args.retrieval_dir)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, query), 'w') as fout:
        # return max. 1000 objects
        for i, _ in zip(np.argsort(dists), range(1000)):
            # only return objs classified as same class as query
            if lbls[i] == query_lbl:
                print(fnames[i], file=fout)


def eval_shrec17_output(outdir):
    basedir = Path(os.path.realpath(__file__)).parent / '..' / '..'
    evaldir = basedir / 'external/shrec17_evaluator'
    assert basedir.is_dir()
    assert evaldir.is_dir()
    assert os.path.isdir(outdir)
    evaldir = str(evaldir)
    if outdir[-1] != '/':
        outdir += '/'
    # include ~/bin in path
    env = os.environ.copy()
    env['PATH'] = '{}/bin:{}'.format(env['HOME'], env['PATH'])
    p = subprocess.Popen(['node', 'evaluate.js', outdir],
                         cwd=evaldir,
                         env=env)
    p.wait()

    import pandas as pd
    data = pd.read_csv('{}/{}.summary.csv'
                       .format(evaldir, outdir.split('/')[-2]))

    return data


def eval_retrieval(descriptors, gt_labels, pred_labels, fnames, eval_shrec=False):
    logger.info('Evaluating retrieval...')
    # compute pairwise distances
    dists = squareform(pdist(descriptors, 'cosine'))
    logger.info('Computed pairwise distances between {} samples'
                .format(len(dists)))
    fnames = [Path(f).parts[-2] for f in fnames]
    retrieval_out = dict(mAP_inst=[], mAP_cls={})
    for d, c, l, f in zip(dists, gt_labels, pred_labels, fnames):
        positive = gt_labels == c
        if args.retrieval_include_same:
            # set dists to 0 of elements from the same class to always return them
            # TODO: should sort by distance, but it won't affect mAP
            d = d.copy()
            d[pred_labels == c] = 0

        # save list of retrieved elements in order
        if eval_shrec:
            save_shrec17_output(f, d, fnames, l, pred_labels)
        score = 100. * average_precision_score(positive, 2 - d)
        retrieval_out['mAP_inst'].append(score)
        retrieval_out['mAP_cls'][c] = retrieval_out['mAP_cls'].get(c, [])
        retrieval_out['mAP_cls'][c].append(score)
        # retrieval_out['AUC'].append(roc_auc_score(positive, 2-d))
    # compute per class avg
    retrieval_out['mAP_cls'] = [np.mean(v)
                                for v in retrieval_out['mAP_cls'].values()]

    if eval_shrec:
        logger.info('Evaluating SHREC17')
        out = eval_shrec17_output(str((Path(args.logdir) / 'retrieval')
                                      .absolute()))

        def totable(x): return '|{}|{}|'.format(os.path.basename(args.logdir),
                                                '|'.join([str(xx) for xx in x]))
        logger.info(totable(out.columns))
        logger.info(totable(out.values[0]))
        logger.info(totable(out.values[-1]))

    return {k: np.mean(v) for k, v in retrieval_out.items()}


# Validation and Testing
def eval(data_loader, device, model, criterion, eval_shrec):
    def _acc(preds, tgts):
        return 100. * sum(preds == tgts) / len(preds)

    # introspected features from the model
    introspection = []
    # Eval
    if args.eval_retrieval:
        def hook(model, input, output):
            introspection.append({'descriptor': output.detach().cpu().numpy()})
        model.final_descriptor.register_forward_hook(hook)

    # save fmaps
    if args.save_fmaps:
        def hook(model, input, output):
            introspection.append(
                {'initial_group': output.detach().cpu().numpy()})
        model.initial_group.register_forward_hook(hook)

        def hook(model, input, output):
            introspection.append({'fmaps': output.detach().cpu().numpy()})
        for l in model.gc_layers:
            l.register_forward_hook(hook)

    losses = []
    preds = []
    tgts = []
    for i, (inputs, targets, fnames, tload) in enumerate(data_loader):
        with torch.no_grad():
            inputs, targets = load_inputs_targets(inputs, targets, device)

            # compute output
            outputs = model(inputs)

            if args.label_smoothing > 0:
                onehot_targets = smooth_labels(targets, device)
                loss = criterion(outputs, onehot_targets)
            else:
                loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            losses.append(loss.cpu().numpy())
            preds.append(predicted.cpu().numpy())
            tgts.append(targets.cpu().numpy())

            # accumulate descriptors for retrieval
            if args.eval_retrieval:
                introspection[-1]['gt_label'] = targets.cpu().numpy()
                introspection[-1]['pred_label'] = predicted.cpu().numpy()
                introspection[-1]['fname'] = fnames

            # these are huge!
            if args.save_fmaps:
                # saving only view 0
                introspection[-1]['inputs'] = inputs.cpu().numpy()[:, 0]

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f  Acc: %.4f" %
                  (i + 1,
                   len(data_loader),
                   losses[-1],
                   _acc(preds[-1], tgts[-1])))

        if args.max_steps > 0 and i > args.max_steps:
            break

    # save / eval descriptors
    retrieval_dict = util.ld2dl(introspection)
    if args.eval_retrieval:
        retrieval_dict = {k: np.concatenate(v)
                          for k, v in retrieval_dict.items()}
        retrieval_out = eval_retrieval(retrieval_dict['descriptor'],
                                       retrieval_dict['gt_label'],
                                       retrieval_dict['pred_label'],
                                       retrieval_dict['fname'],
                                       eval_shrec)
        # np.savez('/tmp/desc.npz', retrieval_dict)
    else:
        retrieval_out = {}

    if args.save_fmaps:
        outfile = str(Path(args.logdir) / 'introspection.npz')
        print('saving {}'.format(outfile))
        np.savez(outfile, retrieval_dict)

    preds, tgts = np.concatenate(preds), np.concatenate(tgts)

    metrics = retrieval_out
    metrics['acc_inst'] = _acc(preds, tgts)
    metrics['acc_cls'] = [_acc(preds[tgts == c], tgts[tgts == c])
                          for c in np.unique(tgts)]
    logger.info('acc per class={}'.format(metrics['acc_cls']))
    metrics['acc_cls'] = np.mean(metrics['acc_cls'])
    metrics['loss'] = np.mean(losses)

    # compute combined score (avg between acc / map micro and macro)
    metrics['combined'] = np.mean([metrics.get(s, 0) for s in
                                   ['acc_inst', 'acc_cls', 'mAP_inst', 'mAP_cls']])

    return metrics, inputs


def init_train_state():
    return dict(start_epoch=0,
                lr=args.lr,
                step=0,
                steps_epoch=0)


def get_summary_ims(inputs):
    shp = inputs.shape
    if len(shp) == 5:
        nsplits = 10 if args.n_homogeneous == 20 else 12
        ims = torch.cat(torch.cat(
            torch.split(inputs[0], nsplits), dim=2).unbind(),
            dim=2)
    else:
        ims = inputs[0]

    return ims


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    logger.info('Loading data...')
    train_loader, val_loader, classes = custom_dataset.load_data(args)

    # override autodetect if n_classes is given
    if args.n_classes > 0:
        classes = np.arange(args.n_classes)

    model = load_model(classes)

    logger.info('Loaded model; params={}'.format(util.count_parameters(model)))
    if not args.cpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    model.to(device)
    cudnn.benchmark = True
    logger.info('Running on ' + str(device))

    summary_writer = Logger(args.logdir)

    # Loss and Optimizer
    n_epochs = args.epochs
    if args.label_smoothing > 0:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_state = init_train_state()
    # freeze layers
    for l in args.freeze_layers:
        for p in getattr(model, l).parameters():
            p.requires_grad = False
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train_state['lr'],
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'nesterov':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=train_state['lr'],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    # this is used to warm-start
    if args.warm_start_from:
        logger.info('Warm-starting from {}'.format(args.warm_start_from))
        assert os.path.isfile(args.warm_start_from)
        train_state = load_checkpoint(args.warm_start_from, model, optimizer)
        logger.info('Params loaded.')
        # do not override train_state these when warm staring
        train_state = init_train_state()

    ckptfile = str(Path(args.logdir) / args.latest_fname)
    if os.path.isfile(ckptfile):
        logger.info('Loading checkpoint: {}'.format(ckptfile))
        train_state = load_checkpoint(ckptfile, model, optimizer)
        logger.info('Params loaded.')
    else:
        logger.info('Checkpoint {} not found; ignoring.'.format(ckptfile))

    # Training / Eval loop
    epoch_time = []                 # store time per epoch
    # we save epoch+1 to checkpoints; but for eval we should repeat prev. epoch
    if args.skip_train:
        train_state['start_epoch'] -= 1
    for epoch in range(train_state['start_epoch'], n_epochs):
        logger.info('Epoch: [%d/%d]' % (epoch + 1, n_epochs))
        start = time.time()

        if not args.skip_train:
            model.train()
            train(train_loader, device, model, criterion, optimizer, summary_writer, train_state,
                  n_classes=len(classes))
            logger.info('Time taken: %.2f sec...' % (time.time() - start))
            if epoch == 0:
                train_state['steps_epoch'] = train_state['step']
        # always eval on last epoch
        if not args.skip_eval or epoch == n_epochs - 1:
            logger.info('\n Starting evaluation...')
            model.eval()
            eval_shrec = True if epoch == n_epochs - 1 and args.retrieval_dir else False
            metrics, inputs = eval(
                val_loader, device, model, criterion, eval_shrec)

            logger.info('\tcombined: %.2f, Acc: %.2f, mAP: %.2f, Loss: %.4f' %
                        (metrics['combined'],
                         metrics['acc_inst'],
                         metrics.get('mAP_inst', 0.),
                         metrics['loss']))

            # Log epoch to tensorboard
            # See log using: tensorboard --logdir='logs' --port=6006
            ims = get_summary_ims(inputs)
            if not args.nolog:
                util.logEpoch(summary_writer, model, epoch + 1, metrics, ims)
        else:
            metrics = None

        # Decaying Learning Rate
        if args.lr_decay_mode == 'step':
            if (epoch + 1) % args.lr_decay_freq == 0:
                train_state['lr'] *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = train_state['lr']

        # Save model
        if not args.skip_train:
            logger.info('\tSaving latest model')
            util.save_checkpoint({
                'epoch': epoch + 1,
                'step': train_state['step'],
                'steps_epoch': train_state['steps_epoch'],
                'state_dict': model.state_dict(),
                'metrics': metrics,
                'optimizer': optimizer.state_dict(),
                'lr': train_state['lr'],
            },
                str(Path(args.logdir) / args.latest_fname))

        total_epoch_time = time.time() - start
        epoch_time.append(total_epoch_time)
        logger.info('Total time for this epoch: {} s'.format(total_epoch_time))

        # if last epoch, show eval results
        if epoch == n_epochs - 1:
            logger.info(
                '|model|combined|acc inst|acc cls|mAP inst|mAP cls|loss|')
            logger.info('|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.4f}|'
                        .format(os.path.basename(args.logdir),
                                metrics['combined'],
                                metrics['acc_inst'],
                                metrics['acc_cls'],
                                metrics.get('mAP_inst', 0.),
                                metrics.get('mAP_cls', 0.),
                                metrics['loss']))

        if args.skip_train:
            # if evaluating, run it once
            break

        if time.perf_counter() + np.max(epoch_time) > start_time + args.exit_after:
            logger.info('Next epoch will likely exceed alotted time; exiting...')
            break


if __name__ == '__main__':
    main()
