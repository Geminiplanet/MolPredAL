# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
import os
import random
import argparse
import time

import torch.cuda
from torch import optim, nn
from torch_geometric.loader import DataLoader

from dataset import load_dataset_random

from config import *
from models.model import TrimNet
from models.query_models import LossNet
# from train_test import train
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='../data/', help="all data dir")
parser.add_argument("--dataset", type=str, default='bace', help="muv,tox21,toxcast,sider,clintox,hiv,bace,bbbp")
parser.add_argument('--seed', default=68, type=int)
parser.add_argument("--gpu", type=int, nargs='+', default=0, help="CUDA device ids")

parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
parser.add_argument('--heads', default=4, type=int)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument('--lr_scheduler_patience', default=10, type=int)
parser.add_argument('--early_stop_patience', default=-1, type=int)
parser.add_argument('--lr_decay', default=0.98, type=float)
parser.add_argument('--focalloss', default=False, action="store_true")

parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument("--exps_dir", default='test/', type=str, help="out dir")
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--load', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':


    args.tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
             'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    # load training and testing dataset
    data_train, data_val, data_test = load_dataset_random('data/', 'tox21', args.seed, args.tasks)
    print(f'train: {data_train}, val: {data_val}, test: {data_test}')
    data_unlabeld = data_train
    NUM_TRAIN = len(data_train)
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:ADDENNUM]
    unlabelled_set = indices[ADDENNUM:]
    train_loader = DataLoader(data_train, batch_size=BATCH, shuffle=True)
    # for data in train_loader:
    #     print(data.y)
    #     for i in range(12):
    #         y_label = data.y[:, i].squeeze()
    #         print(y_label)
    test_loader = DataLoader(data_test, batch_size=BATCH)
    # data_loaders = {'train': train_loader, 'test': test_loader}

    args.parallel = True if args.gpu and len(args.gpu) > 1 else False
    args.parallel_devices = args.gpu
    args.tag = time.strftime("%m-%d-%H-%M") if args.exp_name is None else args.exp_name
    args.exp_path = os.path.join(args.exps_dir, args.tag)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    args.code_file_path = os.path.abspath(__file__)
    args.out_dim = 2 * len(args.tasks)
    option = args.__dict__


    for cycle in range(1):
        # # model: create new instance for every cycle so that it resets
        # with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #     resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
        #     loss_module = LossNet().cuda()
        #
        # models = {'backbone': resnet18, 'module': loss_module}
        # torch.backends.cudnn.benchmark = True

        # # loss, criterion and scheduler (re)initialization
        # criterion = nn.CrossEntropyLoss(reduction='none')
        # optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        # sched_backbone = optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        # optim_module = optim.SGD(models['module'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        # sched_module = optim.lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
        # optimizers = {'backbone': optim_backbone, 'module': optim_module}
        # schedulers = {'backbone': sched_backbone, 'module': sched_module}

        # trainging and testing
        in_dim = data_train.num_node_features
        edge_in_dim = data_train.num_edge_features
        weight = data_train.weights

        if not args.eval:
            model = TrimNet(in_dim, edge_in_dim, hidden_dim=option['hid'], depth=option['depth'],
                          heads=option['heads'], dropout=option['dropout'], outdim=option['out_dim'])
            trainer = Trainer(option, model, data_train, data_val, data_test, weight=weight, tasks_num=len(args.tasks))
            trainer.train()
            print('Testing...')
            trainer.load_best_ckpt()
            trainer.valid_iterations(mode='eval')
        # else:
        #     ckpt = torch.load(args.load)
        #     option = ckpt['option']
        #     model = TrimNet(option['in_dim'], option['edge_in_dim'], hidden_dim=option['hid'], depth=option['depth'],
        #                   heads=option['heads'], dropout=option['dropout'], outdim=option['out_dim'])
        #     if not os.path.exists(option['exp_path']): os.makedirs(option['exp_path'])
        #     model.load_state_dict(ckpt['model_state_dict'])
        #     model.eval()
        #     trainer = Trainer(option, model, data_train, data_val, data_test, weight=weight,
        #                       tasks_num=len(args.tasks))
        #     trainer.valid_iterations(mode='eval')









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
