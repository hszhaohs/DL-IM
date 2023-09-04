import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import models
from utils.data_spec import load_dataset, generate_dataset, DataSetGeneratorSpec
from utils.utils import Logger
from utils import record
import logging
from trainer import train_model, test_model

import argparse
import ruamel.yaml as yaml
from easydict import EasyDict


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Kaguya Inversion 1DCNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default='0', type=str)
    parser.add_argument('--config', default='./configs.yaml', type=str, help='config file path')
    parser.add_argument('--model-type', default='OneDCNN', type=str)
    parser.add_argument('--dataname', default='TiO2', type=str,
                        choices=['TiO2', 'FeO', 'Al2O3', 'MgO', 'CaO', 'SiO2'])
    parser.add_argument('--do-scale', default=True, type=bool)
    parser.add_argument('--epochs', default=100, type=int, help='epochs (default: 80)')
    parser.add_argument('--batch-size', default=54, type=int, help='batch size (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001)')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay (default: 0.0001)')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--datapath', default='./datasets/', type=str)
    parser.add_argument('--loss-type', default='MSE', type=str, choices=['MSE', 'L1'])
    parser.add_argument('--load-from', default=None, type=str, help='load-from')
    parser.add_argument('--resume-from', default=None, type=str, help='resume-from')
    parser.add_argument('--patience', default=50, type=int, help='patience (default: 30)')
    parser.add_argument('--save-ckpt', default=True, type=bool)
    parser.add_argument('--record-type', default='CV', type=str, choices=['CV'])
    
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_model(cfg):
    if cfg.model.model_type == 'OneDCNN':
        model_init = models.OneDCNN().to(cfg.device)

    return model_init


args = parse_args()
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)

config['model']['model_type'] = args.model_type
config['loss']['loss_type'] = args.loss_type
config['data']['dataname'] = args.dataname
config['data']['do_scale'] = args.do_scale

config['optim']['base_lr'] = args.lr
config['optim']['weight_decay'] = args.wd
config['optim']['patience'] = args.patience
config['epochs'] = args.epochs
config['batch_size'] = args.batch_size
config['save_ckpt'] = args.save_ckpt
config['record_type'] = args.record_type
if args.load_from is not None:
    config['load_from'] = args.load_from
if args.resume_from is not None:
    config['resume_from'] = args.resume_from

cfg = EasyDict(config)

if torch.cuda.is_available():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    cudnn.benchmark = True

device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
cfg.device = device

cfg.work_dir = './log_{}_{}_Epoch={}_{}'.format(cfg.model.model_type, cfg.data.dataname, cfg.epochs, timestamp)
if not os.path.exists(cfg.work_dir):
    os.makedirs(cfg.work_dir)
with open(os.path.join(cfg.work_dir, 'config.yaml'), 'w') as ff:
    yaml.dump(config, ff, Dumper=yaml.RoundTripDumper)

# for Monte Carlo runs
# seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
logger = Logger(log_file_name=log_file, log_level=logging.DEBUG, logger_name="OneDCNN").get_log()

logger.info('-----Importing Dataset-----')
logger.info(' ** Dataset\'s name: {}'.format(cfg.data.dataname))
if cfg.data.do_scale:
    spec_data, gt, cfg.scaler_X, cfg.scaler_Y = load_dataset(cfg.data.dataname, cfg.data.datapath, do_scale=cfg.data.do_scale)
else:
    spec_data, gt = load_dataset(cfg.data.dataname, cfg.data.datapath, do_scale=cfg.data.do_scale)
num_samples = len(gt)

logger.info(' ** Dataset\'s shape: {}'.format(spec_data.shape))
logger.info(' ** Model\'s Type is: {}'.format(cfg.model.model_type))
cfg.logger = logger

Pred_CV = []
for idx_sample in range(num_samples):
    setup_seed(args.seed)
    cfg.latest_ckpt = os.path.join(cfg.work_dir, 'model_{}_weights_latest_IdxCV={}.pth'.format(cfg.data.dataname,
                                                                                               idx_sample+1))
    cfg.best_ckpt = os.path.join(cfg.work_dir, 'model_{}_weights_best_IdxCV={}.pth'.format(cfg.data.dataname,
                                                                                           idx_sample+1))

    cfg.logger.info('-------- Train model: Num_Samples {}--------'.format(idx_sample+1))
    x_train, y_train, x_test, y_test = generate_dataset(spec_data, gt, idx_sample, num_samples)

    train_dataset = DataSetGeneratorSpec(x_train, targets=y_train)
    train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                       shuffle=cfg.data.is_shuffle, num_workers=cfg.num_workers)
    test_dataset = DataSetGeneratorSpec(x_test, targets=y_test)
    test_dataloader = data.DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # initializing model
    model = create_model(cfg)

    # criterion
    if cfg.loss.loss_type == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean').to(cfg.device)
    elif cfg.loss.loss_type == 'L1':
        criterion = torch.nn.L1Loss(reduction='mean').to(cfg.device)
    # optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                        patience=10, min_lr=0, verbose=True)

    # train CV model
    dataloaders = {'train': train_dataloader, 'valid': test_dataloader}
    tic1 = time.perf_counter()
    model_ft, _ = train_model(model, dataloaders, criterion, optimizer, lr_scheduler, cfg)
    toc1 = time.perf_counter()

    # test CV model
    cfg.logger.info('\n-------- Test Model --------')
    tic2 = time.perf_counter()
    test_rmse, test_pred = test_model(model_ft, test_dataloader, criterion, cfg)
    toc2 = time.perf_counter()
    Pred_CV.append(test_pred)

R2_score_CV_np, RMSE_CV_np = record.record_output(gt, Pred_CV, cfg)

Metric_all = np.vstack([RMSE_CV_np, R2_score_CV_np])
Metric_index = ['RMSE_CV', 'R2_CV']
Metric_col = ['Result_{}'.format(cfg.data.dataname)]
Metric_df = pd.DataFrame(Metric_all, index=Metric_index, columns=Metric_col)
Metric_file = os.path.join(cfg.work_dir, 'Records_RMSE-R2_{}.csv'.format(cfg.data.dataname))
Metric_df.to_csv(Metric_file)

print(Metric_df)

cfg.logger.info('--------Finished-----------')


