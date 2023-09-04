import copy
import time
import math
import shutil
import torch
from tqdm import tqdm
from utils.utils import adjust_learning_rate, get_current_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, latest_path='latest.pth', best_path='best.pth'):
    torch.save(state, latest_path)
    if is_best:
        shutil.copyfile(latest_path, best_path)


def train(model, dataloader, criterion, optimizer, lr_scheduler, epoch, cfg):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    rmse1 = AverageMeter()
    
    # Set model to training mode
    model.train()
    
    # Use tqdm for iterating through data
    iters_pre_epoch = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=iters_pre_epoch)
    for iters, (inputs, targets) in pbar:
        # prepare data
        inputs = inputs.to(cfg.device)
        targets = targets.to(cfg.device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        
        if cfg.model.model_type != 'OneDCNN':
            lr_scheduler.step(epoch + iters / iters_pre_epoch)
        
        # measure error and record loss
        losses.update(loss.item(), inputs.size(0))
        rmse_batch = math.sqrt(loss.item())
        rmse1.update(rmse_batch, inputs.size(0))
        pbar.set_description('Train -> - loss: {loss.avg:.6f} - rmse: {rmse1.avg:.6f}'.format(loss=losses, rmse1=rmse1))
        
    cfg.logger.info('Train -> -     loss: {loss.avg:.6f} -     rmse: {rmse1.avg:.6f}'.format(
                    loss=losses, rmse1=rmse1))
    
    return rmse1.avg, losses.avg


def validate(model, dataloader, criterion, cfg):
    """
        Run evaluation
    """
    losses = AverageMeter()
    rmse1 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    # Use tqdm for iterating through data
    for inputs, targets in dataloader:
        # prepare data
        inputs = inputs.to(cfg.device)
        targets = targets.to(cfg.device)
        
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure error and record loss
        losses.update(loss.item(), inputs.size(0))
        rmse_batch = math.sqrt(loss.item())
        rmse1.update(rmse_batch, inputs.size(0))
        
    cfg.logger.info('Valid -> - val_loss: {loss.avg:.6f} - rmse: {rmse1.avg:.6f}'.format(loss=losses, rmse1=rmse1))
    
    return rmse1.avg, losses.avg


def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, cfg):
    """
        Run train for all epoch
    """
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_rmse = 1e4
    patience = 0
    
    for epoch in range(cfg.epochs):
        lr = get_current_lr(optimizer)[-1]
        cfg.logger.info('\n')
        cfg.logger.info(' -- running {}-{}   Epoch {:03d}/{:03d}   lr:{:.6f} -- '.format(
            cfg.data.dataname, cfg.model.model_type, epoch+1, cfg.epochs, lr))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            dataloader = dataloaders[phase]
            if phase == 'train':
                train_rmse, train_loss = train(model, dataloader, criterion, optimizer, lr_scheduler, epoch, cfg)
            else:
                val_rmse, val_loss = validate(model, dataloader, criterion, cfg)
                if cfg.model.model_type == 'OneDCNN':
                    lr_scheduler.step(val_loss)

        is_best = val_rmse < best_rmse
        if is_best:
            cfg.logger.info('val_rmse improved from {:.6f} to {:.6f}, saving model...'.format(best_rmse, val_rmse))
            patience = 0
            best_rmse = val_rmse
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            cfg.logger.info('val_rmse did not improve from {:.6f} at patience={:02d}'.format(best_rmse, patience))

        cfg.logger.info('Best Valid RMSE: {:.6f}'.format(best_rmse))
            
        if cfg.save_ckpt:
            dict_checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_rmse': best_rmse,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(dict_checkpoint, is_best, latest_path=cfg.latest_ckpt, best_path=cfg.best_ckpt)
        
        if patience == cfg.optim.patience:
            cfg.logger.info('\n**************** Early Stopping ****************\n')
            break
        
    time_elapsed = time.time() - since
    
    cfg.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    cfg.logger.info('Best Val RMSE: {:.6f}'.format(best_rmse))
    
    model_best = copy.deepcopy(model)
    model_best.load_state_dict(best_model_wts)
    
    return model_best, model


def test_model(model, dataloader, criterion, cfg):
    """
        Run evaluation
    """
    rmse1 = AverageMeter()
    pred_all = []
    
    # switch to evaluate mode
    model.eval()
    # Use tqdm for iterating through data
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for ii, data in pbar:
        # prepare data
        inputs, targets = data
        inputs = inputs.to(cfg.device)
        targets = targets.to(cfg.device)
        
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        rmse_batch = math.sqrt(loss.item())
        rmse1.update(rmse_batch, inputs.size(0))
        
        pred = outputs.detach().cpu().numpy()
        pred_all.extend(list(pred[:, 0]))
        
        # compute computation time and loss and err
        pbar.set_description('Test  -> - test_rmse: {rmse1.avg:.6f}'.format(rmse1=rmse1))

    cfg.logger.info('Test  -> - test_rmse: {rmse1.avg:.6f}'.format(rmse1=rmse1))
    
    return [rmse1.avg], [pred_all]


def pred_model(model, dataloader, cfg):
    """
        Run prediction
    """
    pred_all = []

    # switch to evaluate mode
    model.eval()
    for ii, data in enumerate(dataloader):
        # prepare data
        inputs = data
        inputs = inputs.to(cfg.device)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)

        pred = outputs.detach().cpu().numpy()
        pred_all.extend(list(pred[:, 0]))

    return pred_all



