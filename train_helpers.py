import sys

sys.path.append('..')
from utils.dataset_utils import prepare_data_loaders
import segmentation.tiramisu.tiramisu as tiramisu
from metrics.metrics_utils import TensorboardLogger, MetricsManager

import torch
import torch.optim

from collections import OrderedDict
import json
from os.path import join, exists, isfile
from os import makedirs
import shutil
import time
from tqdm import tqdm


def load_hparams(hparams_path):
    if not exists(hparams_path):
        raise Exception('You must provide path to existing .json file with hyperparameters')

    with open(hparams_path, 'r') as f:
        hparams = json.load(f, object_pairs_hook=OrderedDict)

    return hparams


def create_logdirs(modeldir):
    if not exists(modeldir):
        makedirs(modeldir)

    cur_time_str = time.ctime().replace(' ', '_').replace(':', '-')
    tb_dir = join(modeldir, 'logs', cur_time_str)
    runs_dir = join(modeldir, 'runs', cur_time_str)

    return tb_dir, runs_dir


def prepare_training(hparams, num_classes):
    if 'model_params' not in hparams:
        raise Exception('You must add model params to hparams')

    model = tiramisu.__dict__[hparams['model_params']['model']](num_classes)

    if 'criterion_params' not in hparams or \
            'criterion' not in hparams['criterion_params']:
        raise Exception('You must add criterion params to hparams')

    criterion_params = hparams['criterion_params']
    criterion_name = criterion_params.pop('criterion')
    weights_path = criterion_params.pop('weights_path')

    with open(weights_path, 'r') as infile:
        criterion_params["weight"] = torch.Tensor(json.load(infile))

    criterion = torch.nn.__dict__[criterion_name](**criterion_params)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    if 'optimizer_params' not in hparams or \
            'optimizer' not in hparams['optimizer_params']:
        raise Exception('You must add optimizer params to hparams')

    optimizer_params = hparams['optimizer_params']
    optimizer_name = optimizer_params.pop('optimizer')
    optimizer = torch.optim.__dict__[optimizer_name](
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_params
    )

    if 'scheduler_params' in hparams:
        scheduler_params = hparams['scheduler_params']
        if 'scheduler' not in scheduler_params:
            raise Exception('If you provided scheduler params you also must add scheduler name')
        scheduler_name = scheduler_params.pop('scheduler')

        scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](
            optimizer, **scheduler_params
        )
    else:
        scheduler = None

    return model, criterion, optimizer, scheduler


def run_train_val_loader(metrics_manager, epoch, loader, mode, model,
                         criterion, optimizer):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(batch['image'].cuda(async=True), requires_grad=False)
        else:
            input_var = torch.autograd.Variable(batch['image'], requires_grad=False)

        if torch.cuda.is_available():
            target_var = torch.autograd.Variable(batch['mask'].type(torch.int64).cuda(async=True), requires_grad=False)
        else:
            target_var = torch.autograd.Variable(batch['mask'].type(torch.int64), requires_grad=False)
        target_var = target_var.squeeze()

        with torch.set_grad_enabled(mode == 'train'):
            output = model.forward(input_var)
            loss = criterion(output, target_var)
            loss_val = float(loss.data.cpu().numpy())

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics_manager.update_epoch_train_metrics(
                    loss_val, output.data, target_var.data
                )
            else:
                metrics_manager.update_epoch_valid_metrics(
                    loss_val, output.data, target_var.data
                )

    train_loss, valid_loss = metrics_manager.get_cur_loss()
    if mode == 'train':
        epoch_metrics_str = "loss\t{:.3f}".format(train_loss)
    else:      
        epoch_metrics_str = "loss\t{:.3f}".format(valid_loss)
    print("{epoch} * Epoch ({mode}): ".format(epoch=epoch, mode=mode), epoch_metrics_str)

    return metrics_manager.get_cur_loss()


def save_checkpoint(model, optimizer, metrics_manager, epoch, hparams, is_best, logdir):
    train_metrics_history, valid_metrics_history, \
    best_loss, best_metrics = metrics_manager.get_metrics_history()

    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "train_metrics_history": train_metrics_history,
        "val_metrics_history": valid_metrics_history,
        "model": model,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer,
        "optimizer_state_dict": optimizer.state_dict(),
        "hparams": hparams
    }

    if not exists(logdir):
        makedirs(logdir)

    filename = "{}/checkpoint.pth.tar".format(logdir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/checkpoint.best.pth.tar'.format(logdir))


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    best_metrics = checkpoint['best_metrics']
    train_metric_history = checkpoint['train_metrics_history']
    val_metric_history = checkpoint['val_metrics_history']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    return start_epoch, best_loss, best_metrics, \
           train_metric_history, val_metric_history, \
           model_state_dict, optimizer_state_dict


def run_train(args):
    hparams = load_hparams(args.hparams)
    tb_dir, runs_dir = create_logdirs(args.logdir)

    train_loader = prepare_data_loaders(hparams, 'train')
    valid_loader = prepare_data_loaders(hparams, 'valid')

    model, criterion, optimizer, scheduler = prepare_training(hparams, 2)
    metrics_manager = MetricsManager(hparams, 2)

    start_epoch = 0

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            start_epoch, best_loss, best_metrics, \
            train_metric_history, val_metric_history, \
            model_state_dict, optimizer_state_dict = load_checkpoint(args.resume)

            metrics_manager.init_with_history(train_metric_history, val_metric_history,
                                              best_loss, best_metrics)

            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model).cuda()
                criterion = criterion.cuda()

            optimizer.load_state_dict(optimizer_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch - 1))
        else:
            raise Exception("no checkpoint found at '{}'".format(args.resume))
    else:
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()

    tb_logger = TensorboardLogger(metrics_manager, tb_dir, {"трава у дома": 0, "дом": 1})

    if 'training_params' not in hparams:
        raise Exception('You must provide training_params in hparams')

    training_params = hparams['training_params']
    if 'epochs' not in training_params or 'batch_size' not in training_params:
        raise Exception('You must add epochs and batch_size parameters into hparams')

    print('Training started')

    for epoch in range(start_epoch, training_params['epochs']):
        metrics_manager.add_epoch()
        train_loss = run_train_val_loader(metrics_manager, epoch, train_loader, 'train',
                                          model, criterion, optimizer)
        valid_loss = run_train_val_loader(metrics_manager, epoch, valid_loader, 'valid',
                                          model, criterion, optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        if metrics_manager.is_current_best():
            save_checkpoint(model, optimizer, metrics_manager, epoch,
                            hparams, is_best=True, logdir=runs_dir)

        tb_logger.update_metrics(epoch)

    tb_logger.close()
