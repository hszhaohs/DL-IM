import os
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import models
from utils.data_spec import load_dataset, DataSetGeneratorSpec
from utils import record
from trainer import pred_model

import argparse
import ruamel.yaml as yaml
from easydict import EasyDict
from glob import glob
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Kaguya Inversion 1DCNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default='0', type=str)
    parser.add_argument('--dataname', default='TiO2', type=str, choices=['TiO2', 'FeO', 'Al2O3', 'MgO', 'CaO', 'SiO2'])
    parser.add_argument('--datapath', default='./spec_Kaguya_MI_by_row', type=str)
    
    return parser.parse_args()


def create_model(cfg):
    if cfg.model.model_type == 'OneDCNN':
        model_init = models.OneDCNN().to(cfg.device)

    return model_init


args = parse_args()
args.config = './model_ckpt/log_{}/config.yaml'.format(args.dataname)
args.load_from = './model_ckpt/log_{}/model_best.pth'.format(args.dataname)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)

config['data']['datapath_pred'] = args.datapath
config['load_from'] = args.load_from

cfg = EasyDict(config)

if torch.cuda.is_available():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    cudnn.benchmark = True

device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
cfg.device = device

cfg.work_dir = './pred_{}_{}'.format(cfg.data.dataname, timestamp)
if not os.path.exists(cfg.work_dir):
    os.makedirs(cfg.work_dir)

# model
model_best = create_model(cfg)
ckpt = torch.load(cfg.load_from, map_location=cfg.device)
model_best.load_state_dict(ckpt['state_dict'])
model_best.to(cfg.device)

# Load train data for Scaler info
if cfg.data.do_scale:
    _, _, cfg.scaler_X, cfg.scaler_Y = load_dataset(cfg.data.dataname, cfg.data.datapath, do_scale=cfg.data.do_scale)

print('-------- Predicting --------')
# Load spec data for pred
spec_list = glob(os.path.join(cfg.data.datapath_pred, 'spec*.csv'))

pbar = tqdm(enumerate(spec_list), total=len(spec_list))
for ii_spec, spec_file in pbar:
    pbar.set_description('predicting {} '.format(os.path.basename(spec_file)))
    cfg.filename = spec_file
    # Load spec data for prediction
    spec_df = pd.read_csv(spec_file)
    spec_np = spec_df.to_numpy()
    if cfg.data.dataname in ['FeO', 'TiO2']:
        spec_data = spec_np[:, 1:].astype(np.float32)
    elif cfg.data.dataname in ['Al2O3', 'MgO', 'CaO', 'SiO2']:
        spec_data = spec_np[:, 1:-2].astype(np.float32)
    num_samples = len(spec_data)
    cfg.batch_size = num_samples
    if cfg.data.do_scale:
        spec_data = cfg.scaler_X.transform(spec_data)
    spec_dataset = DataSetGeneratorSpec(spec_data)
    spec_dataloader = data.DataLoader(spec_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # pred
    pred_result = pred_model(model_best, spec_dataloader, cfg)

    record.record_pred(pred_result, cfg)

print('--------  Finished  --------')

