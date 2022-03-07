import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import random

def setup_seed(seed):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader',module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch',module_arch)

    # get function handles of loss and metrics
    loss_class = getattr(module_loss, config["loss"]["type"])
    criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer',torch.optim,model.parameters())

    if "type" in config._config["lr_scheduler"]:
        lr_scheduler_args = config["lr_scheduler"]["args"]
        gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
        print("step1, step2, warmup_epoch, gamma:",(lr_scheduler_args["step1"],lr_scheduler_args["step2"],lr_scheduler_args["warmup_epoch"],gamma))

        def lr_lambda(epoch):
            if epoch >= lr_scheduler_args["step2"]:
                lr = gamma*gamma
            elif epoch >= lr_scheduler_args["step1"]:
                lr = gamma
            else:
                lr = 1
            warmup_epoch = lr_scheduler_args["warmup_epoch"]
            if epoch < warmup_epoch:
                lr = lr*float(1+epoch)/warmup_epoch
            return lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    else:
        lr_scheduler = None

    trainer = Trainer(
        model                                   ,
        criterion                               ,
        metrics                                 ,
        optimizer                               ,
        config              = config            ,
        data_loader         = data_loader       ,
        valid_data_loader   = valid_data_loader ,
        lr_scheduler        = lr_scheduler
    )
    setup_seed(39789)
    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c','--config',default=None,type=str,help='config file path (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs','flags type target')
    options = [
        CustomArgs(['--name'],type=str,target='name'),
        CustomArgs(['--save_period'],type=int,target='trainer;save_period'),
        CustomArgs(['--distribution_aware_diversity_factor'],type=float,target='loss;args;additional_diversity_factor'),
        CustomArgs(['--pos_weight'],type=float,target='arch;args;pos_weight'),
        CustomArgs(['--collaborative_loss'],type=int,target='loss;args;collaborative_loss'),
    ]
    config = ConfigParser.from_args(args,options)
    main(config)
