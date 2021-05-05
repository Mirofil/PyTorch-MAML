# python train.py --config=configs/convnet4/mini-imagenet/5_way_1_shot/train_reproduce.yaml --split=sovl --load=False --sotl_freq=5


import argparse
import os
import random
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import wandb

import datasets
import models
import utils
import utils.optimizers as optimizers
from utils import wandb_auth

def main(config, args):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False
  wandb_auth()
  try:
      __IPYTHON__
      wandb.init(project="NAS", group=f"maml")
  except:
      wandb.init(project="NAS", group=f"maml", config=config)

  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')
    ckpt_name += '_{}_way_{}_shot'.format(
      config['train']['n_way'], config['train']['n_shot'])
  if args.tag is not None:
    ckpt_name += '_' + args.tag

  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Dataset #####
  # meta-train
  train_set = datasets.make(config['dataset'], **config['train'])
  utils.log('meta-train set: {} (x{}), {}'.format(
    train_set[0][0].shape, len(train_set), train_set.n_classes))

  # meta-val
  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_classes))
    val_loader = DataLoader(
      val_set, config['val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

  # if args.split == "traintrain" and config.get('val'): # TODO I dont think this is what they meant by train-train :D 
  #   train_set = torch.utils.data.ConcatDataset([train_set, val_set])
  train_loader = DataLoader(
  train_set, config['train']['n_episode'],
  collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)
  
  ##### Model and Optimizer #####

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load') or (args.load is True and os.path.exists(ckpt_path + '/epoch-last.pth')):
    if config.get('load') is None:
      config['load'] = ckpt_path + '/epoch-last.pth'
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']
    model = models.make(config['encoder'], config['encoder_args'],
                        config['classifier'], config['classifier_args'])
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    start_epoch = 1
    max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ##### Training and evaluation #####
    
  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in tqdm(range(start_epoch, config['epoch'] + 1), desc = "Iterating over epochs"):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    
    all_sotls = 0
    all_sovls = 0
    for data_idx, data in enumerate(tqdm(train_loader, desc='meta-train', leave=False)):
      x_shot, x_query, y_shot, y_query = data
      x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
      x_query, y_query = x_query.cuda(), y_query.cuda()

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      if args.split == "traintrain":
        x_query = x_shot
        y_query = y_shot

      logits, sotl = model(x_shot, x_query, y_shot, inner_args, meta_train=True)
      logits = logits.flatten(0, 1)
      labels = y_query.flatten()

      all_sotls += sotl
      
      pred = torch.argmax(logits, dim=-1)
      acc = utils.compute_acc(pred, labels)
      loss = F.cross_entropy(logits, labels)

      all_sovls += loss
      if args.split == "trainval" or (args.split == "sovl" and not data_idx % args.sotl_freq == 0):

        aves['tl'].update(loss.item(), 1)
        aves['ta'].update(acc, 1)
      
        optimizer.zero_grad()
        loss.backward(create_graph=True if args.split == "sovl" else False)
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()
      elif args.split == "traintrain":

        aves['tl'].update(loss.item(), 1)
        aves['ta'].update(acc, 1)

        # sotl = sum(sotl) + loss
        optimizer.zero_grad()
        # sotl.backward()
        loss.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()

      elif args.split == "sotl" and data_idx % args.sotl_freq == 0:
        # TODO doesnt work whatsoever

        aves['tl'].update(loss.item(), 1)
        aves['ta'].update(acc, 1)
        optimizer.zero_grad()
        all_sotls.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()
        all_sotls = 0 # detach
      elif args.split == "sovl" and data_idx % args.sotl_freq == 0:
        # TODO doesnt work whatsoever

        aves['tl'].update(loss.item(), 1)
        aves['ta'].update(acc, 1)
        optimizer.zero_grad()
        all_sovls.backward()
        for param in optimizer.param_groups[0]['params']:
          nn.utils.clip_grad_value_(param, 10)
        optimizer.step()
        all_sovls = 0 # detach

    # meta-val
    if eval_val:
      model.eval()
      np.random.seed(0)

      for data in tqdm(val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits, sotl = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

    # formats output
    log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

    if eval_val:
      log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

    wandb.log({"train_loss": aves['tl'], "train_acc": aves['ta'], "val_loss": aves['vl'], "val_acc": aves['va']})
    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),

      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),

      'training': training,
    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  parser.add_argument('--split', default = "trainval", type=str,
                    help='Whether to do normal MAML or SoTL-MAML')
  parser.add_argument('--load', default = True, type=lambda x: False if x in ["False", "false", "", "None"] else True,
                  help='Whether to do normal MAML or SoTL-MAML')
  parser.add_argument('--sotl_freq', default = 3, type=int,
                help='Whether to do normal MAML or SoTL-MAML')
  
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config, args)