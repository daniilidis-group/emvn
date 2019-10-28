import torch
import os
import logging


def logEpoch(logger, model, epoch, metrics, ims):
    # 1. Log scalar values (scalar summary)
    for tag, value in metrics.items():
        logger.scalar_summary('metrics/' + tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
            logger.histo_summary(
                tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
    info = {'images': ims.cpu().numpy()}
    for tag, images in info.items():
        logger.image_summary(tag, images, epoch)


def logStep(logger, model, epoch,
            loss, accuracy, lr, steps_per_sec,
            ims=None, triplet_loss=None):
    # 1. Log scalar values (scalar summary)
    info = {'train_loss': loss.item(),
            'train_accuracy': accuracy.item(),
            'train_lr': lr,
            'steps_per_sec': steps_per_sec}

    if triplet_loss is not None:
        info['train_triplet_loss'] = triplet_loss

    for tag, value in info.items():
        logger.scalar_summary('train_metrics/' + tag, value, epoch)

    if ims is not None:
        info = {'train_images': ims.cpu().numpy()}
        for tag, images in info.items():
            logger.image_summary(tag, images, epoch)


def save_checkpoint(state, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    torch.save(state, fname)


def init_logger(logdir, fname='logging.log'):
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if fname:
        # create file handler
        logdir = os.path.expanduser(logdir)
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, fname)
        fh = logging.FileHandler(logfile)
        # create console handler
        fh.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)

    return logger


def ld2dl(d):
    """ List of dict to dict of list. """
    if not d:
        return d
    d_out = {k: [] for dd in d for k in dd.keys()}
    for o in d:
        for k, v in o.items():
            d_out[k].append(v)
    return d_out


def count_parameters(model):
    """see https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/11"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
