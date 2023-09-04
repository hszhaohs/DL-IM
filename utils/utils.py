import math
import logging


def get_current_lr(optimizer):
    lr_old = []
    for param_group in optimizer.param_groups:
        lr_old.append(param_group['lr'])
    return lr_old


def adjust_learning_rate(optimizer, epoch, config):
    lr_old = get_current_lr(optimizer)
    lr_new = []
    if config.lr_scheduler.type == 'STEP':
        for lr in lr_old:
            if epoch in config.lr_scheduler.lr_epochs:
                lr *= config.lr_scheduler.lr_mults
            lr_new.append(lr)
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        for lr in lr_old:
            lr = config.lr_scheduler.min_lr + \
                (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
                (1.0 + math.cos(math.pi * ratio)) / 2.0
            lr_new.append(lr)
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        for lr in lr_old:
            lr = config.lr_scheduler.min_lr + \
                (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
                (1.0 - math.tanh(
                    config.lr_scheduler.lower_bound
                    + (config.lr_scheduler.upper_bound
                    - config.lr_scheduler.lower_bound)
                    * ratio)
                ) / 2.0
            lr_new.append(lr)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_new[i]
    return lr_new[-1]


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        #formatter = logging.Formatter(
        #    '[%(asctime)s] - [%(filename)s line:%(lineno)d] : %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s] - : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


