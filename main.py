# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
import random

import torch.cuda
from torch import optim, nn
from torch_geometric.loader import DataLoader

from dataset import load_dataset_random

from config import *
from models import resnet
from models.query_models import LossNet

BATCH = 128
ADDENNUM = 500
CYCLE = 5
NO_CLASSES = 12
CUDA_VISIBLE_DEVICES = 0
LR = 1e-1
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONES = [160, 240]

if __name__ == '__main__':


    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
             'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    # load training and testing dataset
    data_train, data_test = load_dataset_random('data/', 'tox21', 68, tasks)
    print(f'train lens: {len(data_train)}, test lens: {len(data_test)}')
    print(data_train[2].y)
    data_unlabeld = data_train
    NUM_TRAIN = len(data_train)
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:ADDENNUM]
    unlabelled_set = indices[ADDENNUM:]
    train_loader = DataLoader(data_train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=BATCH)

    for cycle in range(CYCLE):
        # model: create new instance for every cycle so that it resets
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
            loss_module = LossNet().cuda()

        models = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = True

        # loss, criterion and scheduler (re)initialization
        criterion = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        sched_backbone = optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
