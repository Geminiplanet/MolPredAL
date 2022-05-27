import random

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from config import *
from dataset import load_tox21_dataset
from models import MolecularVAE, Discriminator, Predictor, LossNet


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for data, _, label in dataloader:
                yield data, label
    else:
        while True:
            for data, _, _ in dataloader:
                yield data


def main():
    # load dataset
    data_train, data_test, data_unlabeled = load_tox21_dataset('data/tox21.csv')
    indices = list(range(data_train.len))
    random.shuffle(indices)
    labeled_set = []
    unlabeled_set = indices
    # train_loader = DataLoader(data_train, batch_size=BATCH, pin_memory=True, drop_last=True)
    # test_loader = DataLoader(data_test, batch_size=BATCH)
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, pin_memory=True)
    # dataloaders = {'train': train_loader, 'test': test_loader}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unlabeled_data = read_data(unlabeled_loader, labels=False)
    # taskModel = Predictor().to(device)
    # loss_module = LossNet().to(device)
    # ranker = loss_module
    # model = {'backbone': taskModel, 'module': loss_module}
    vae = MolecularVAE().to(device)
    discriminator = Discriminator(292).to(device)

    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

    # train vaal
    num_vae_steps = 1
    num_adv_steps = 1
    adversary_param = 1
    beta = 1
    bce_loss = nn.BCELoss()
    train_iterations = len(data_unlabeled) * EPOCHL // BATCH
    vae.train()
    discriminator.train()
    best_acc = 0
    for iter_count in range(20):
        unlabeled_imgs = next(unlabeled_data).float().to(device)

        # VAE step
        for count in range(num_vae_steps):
            # recon, z, mu, logvar = vae(labeled_imgs)
            # unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)

            # labeled_preds = discriminator(mu).squeeze(-1)
            unlabeled_preds = discriminator(unlab_mu).squeeze(-1)

            # lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

            # if self.args.cuda:
            #     lab_real_preds = lab_real_preds.cuda()
            unlab_real_preds = unlab_real_preds.to(device)

            # dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
            #            self.bce_loss(unlabeled_preds, unlab_real_preds)
            dsc_loss = bce_loss(unlabeled_preds, unlab_real_preds)
            # total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
            total_vae_loss = transductive_loss + adversary_param * dsc_loss
            optim_vae.zero_grad()
            total_vae_loss.backward()
            optim_vae.step()

            # # sample new batch if needed to train the adversarial network
            # if count < (self.args.num_vae_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)
            #
            #     if self.args.cuda:
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                # _, _, mu, _ = vae(labeled_imgs)
                _, _, unlab_mu, _ = vae(unlabeled_imgs)

            # labeled_preds = discriminator(mu).squeeze(-1)
            unlabeled_preds = discriminator(unlab_mu).squeeze(-1)

            # lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).to(device)

            # if self.args.cuda:
            #     lab_real_preds = lab_real_preds.cuda()
            #     unlab_fake_preds = unlab_fake_preds.cuda()

            # dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
            #            self.bce_loss(unlabeled_preds, unlab_fake_preds)
            dsc_loss = bce_loss(unlabeled_preds, unlab_fake_preds)
            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            # # sample new batch if needed to train the adversarial network
            # if count < (self.args.num_adv_steps - 1):
            #     labeled_imgs, _ = next(labeled_data)
            #     unlabeled_imgs = next(unlabeled_data)
            #
            #     if self.args.cuda:
            #         labeled_imgs = labeled_imgs.cuda()
            #         unlabeled_imgs = unlabeled_imgs.cuda()
            #         labels = labels.cuda()

        if iter_count % 10 == 9:
            print('Current training iteration: {}'.format(iter_count))
            # print('Current task model loss: {:.4f}'.format(task_loss.item()))
            print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
            print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

    # train and test task model
    labeled_set = indices[:ADDENNUM]
    train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(data_test, batch_size=BATCH)
    task_model = Predictor().to(device)
    loss_module = LossNet().to(device)
    criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(device), reduction='none') for w in data_train.weights]
    optim_task = optim.Adam(task_model.parameters(), lr=0.001, weight_decay=1e-5)
    sched_task = lr_scheduler.MultiStepLR(optim_task, milestones=MILESTONES)
    optim_module = optim.Adam(loss_module.parameters(), lr=LR, weight_decay=WDECAY)
    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
    for cycle in range(CYCLES):
        # train
        for epoch in range(100):
            task_model.train()
            loss_module.train()
            losses = []
            y_label_list = {}
            y_pred_list = {}
            for data in tqdm(train_loader, leave=False, total=len(train_loader)):
                batch_x, _, batch_p = data
                batch_x = batch_x.float().to(device)
                batch_p = batch_p.to(device)
                optim_task.zero_grad()
                _, z, _, _ = vae(batch_x)
                scores = task_model(z)
                target_loss = 0
                # 12 classifier loss
                for i in range(12):
                    y_pred = scores[:, i * 2:(i + 1) * 2]
                    y_label = batch_p[:, i].squeeze().cpu()
                    y_label[np.where(y_label == 6)] = 0
                    y_label = y_label.to(device)
                    # validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                    # if len(validId) == 0:
                    #     continue
                    # y_pred = y_pred[validId]
                    # y_label = y_label[validId]
                    target_loss += criterion[i](y_pred, y_label)
                    y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                    try:
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    except:
                        y_label_list[i] = []
                        y_pred_list[i] = []
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                _, _, feature = vae.encode(batch_x)
                pred_loss = loss_module(feature)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = m_backbone_loss + WEIGHT * m_module_loss
                # loss = target_loss
                loss.backward()
                optim_task.step()
                optim_module.step()
                losses.append(loss.item())
            train_roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(12)]
            train_prc = [metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                                     precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(12)]
            train_loss = np.array(losses).mean()
            train_roc = np.array(train_roc).mean()
            train_prc = np.array(train_prc).mean()
            sched_task.step()
            sched_module.step()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: train loss is {train_loss}, train_roc: {train_roc}, train_prc: {train_prc}')
        # test

        task_model.eval()
        with torch.no_grad():
            y_label_list = {}
            y_pred_list = {}
            for data in tqdm(test_loader, leave=False, total=len(test_loader)):
                batch_x, batch_l, batch_p = data
                batch_x = batch_x.float().to(device)
                # batch_l = batch_l.cuda()
                batch_p = batch_p.to(device)
                _, z, _, _ = vae(batch_x)
                pred = task_model(z)
                for i in range(12):
                    y_pred = pred[:, i * 2:(i + 1) * 2]
                    y_label = batch_p[:, i].squeeze()
                    validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                    if len(validId) == 0:
                        continue
                    y_pred = y_pred[validId]
                    y_label = y_label[validId]
                    y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
                    try:
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    except:
                        y_label_list[i] = []
                        y_pred_list[i] = []
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
            roc = [metrics.roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(12)]
            prc = [metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                               precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(12)]
        print(f'Cycle {cycle + 1}/{CYCLES} || labeled data size {len(labeled_set)}, test roc = {roc}, test prc = {prc}')

        # AL to select data
        # arg = query_samples()


if __name__ == '__main__':
    main()
