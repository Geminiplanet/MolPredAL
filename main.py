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
from data.sampler import SubsetSequentialSampler
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
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss * beta
    return recon_loss + kl_loss


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for data, _, label in dataloader:
                yield data.float(), label
    else:
        while True:
            for data, _, _ in dataloader:
                yield data.float()


def main():
    # load dataset
    data_train, data_test, data_unlabeled = load_tox21_dataset('data/tox21.csv')
    indices = list(range(data_train.len))
    random.shuffle(indices)
    labeled_set = indices[:ADDENNUM]
    unlabeled_set = [x for x in indices if x not in labeled_set]
    train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
    test_loader = DataLoader(data_test, batch_size=BATCH)
    unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = open(f'results_RANDOM_TOX21_main{CYCLES}.txt', 'w')
    # train and test task model
    for cycle in range(CYCLES):
        print('>> Train vae and task model')

        # train vae
        vae = MolecularVAE().to(device)
        optim_vae = optim.Adam(vae.parameters(), lr=2e-5)
        for epoch in range(1, 101):
            vae.train()
            epoch_loss = 0
            for data in tqdm(train_loader, leave=False):
                batch_x, _, _ = data
                batch_x = batch_x.float().to(device)
                optim_vae.zero_grad()
                # !!!
                r = torch.from_numpy(np.zeros((batch_x.shape[0], 1))).type(torch.FloatTensor).to(device)
                recon_x, z_mean, z_logvar = vae(batch_x, r)

                loss = vae_loss(batch_x, recon_x, z_mean, z_logvar, BETA)
                loss.backward()
                epoch_loss += loss
                optim_vae.step()

            if epoch % 20 == 0:
                print(f'epoch {epoch}, vae loss is {epoch_loss / len(train_loader.sampler)}')

        # train task model and loss module
        task_model = Predictor().to(device)
        loss_module = LossNet().to(device)
        criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(device), reduction='none') for w in
                     data_train.weights]
        optim_task = optim.Adam(task_model.parameters(), lr=LR, weight_decay=1e-5)
        sched_task = lr_scheduler.MultiStepLR(optim_task, milestones=MILESTONES)
        optim_module = optim.Adam(loss_module.parameters(), lr=LR, weight_decay=WDECAY)
        sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
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
                optim_module.zero_grad()
                z_mean, _ = vae.encode(batch_x)
                scores, features = task_model(z_mean)
                # 12 classifier loss
                for i in range(12):
                    # target_loss = 0
                    y_pred = scores[:, i * 2:(i + 1) * 2]
                    y_label = batch_p[:, i].squeeze().cpu()
                    y_label[np.where(y_label == 6)] = 0
                    y_label = y_label.to(device)
                    # validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                    # if len(validId) == 0:
                    #     continue
                    # y_pred = y_pred[validId]
                    # y_label = y_label[validId]
                    if i == 0:
                        target_loss = criterion[i](y_pred, y_label)
                    else:
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
                pred_loss = loss_module(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                loss = m_backbone_loss + WEIGHT * m_module_loss
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
                print(
                    f'epoch {epoch}: train loss is {train_loss: .4f}, train_roc: {train_roc: .4f}, train_prc: {train_prc: .4f}')

        # test task model
        print(" >> Test Model")
        task_model.eval()
        loss_module.eval()
        with torch.no_grad():
            y_label_list = {}
            y_pred_list = {}
            for data in tqdm(test_loader, leave=False, total=len(test_loader)):
                batch_x, _, batch_p = data
                batch_x = batch_x.float().to(device)
                batch_p = batch_p.to(device)
                z_mean, _ = vae.encode(batch_x)
                pred, _ = task_model(z_mean)
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
            roc = ('%.4f' % np.array(roc).mean())
            prc = ('%.4f' % np.array(prc).mean())
        print(f'Cycle {cycle + 1}/{CYCLES} || labeled data size {len(labeled_set)}, test roc = {roc}, test prc = {prc}')
        np.array(['RANDOM', cycle + 1, CYCLES, len(labeled_set), roc]).tofile(results, sep=" ")
        results.write("\n")

        if cycle == CYCLES - 1:
            print('Finished.')
            break


        # # AL to select data
        # # Get the indices of the unlabeled samples to train on next cycle
        # unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(unlabeled_set),
        #                               pin_memory=True)
        # labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set),
        #                             pin_memory=True)
        #
        # # train TA-VAAL
        # tavaal_vae = MolecularVAE()
        # discriminator = Discriminator(LATENT_DIM)
        # optim_vaal_vae = optim.Adam(vae.parameters(), lr=5e-4)
        # optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        # ranker = loss_module
        # task_model.eval().to(device)
        # vae.eval().to(device)
        # ranker.eval().to(device)
        # tavaal_vae.train().to(device)
        # discriminator.train().to(device)
        #
        # adversary_param = 1
        # beta = 1
        # num_adv_steps = 1
        # num_vae_steps = 1
        #
        # bce_loss = nn.BCELoss()
        #
        # labeled_data = read_data(labeled_loader)
        # unlabeled_data = read_data(unlabeled_loader)
        #
        # train_iterations = int((ADDENNUM * cycle + len(unlabeled_set)) * 100 / BATCH)
        #
        # for iter_count in range(train_iterations):
        #     labeled_imgs, labels = next(labeled_data)
        #     unlabeled_imgs = next(unlabeled_data)[0]
        #
        #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #         labeled_imgs = labeled_imgs.cuda()
        #         unlabeled_imgs = unlabeled_imgs.cuda()
        #         labels = labels.cuda()
        #     if iter_count == 0:
        #         r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0], 1))).type(
        #             torch.FloatTensor).cuda()
        #         r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0], 1))).type(
        #             torch.FloatTensor).cuda()
        #     else:
        #         with torch.no_grad():
        #             labeled_input, _ = vae.encode(labeled_imgs)
        #             _, features_l = task_model(labeled_input)
        #             unlabeled_input, _ = vae.encode(unlabeled_imgs)
        #             _, feature_u = task_model(unlabeled_input)
        #             r_l = ranker(features_l)
        #             r_u = ranker(feature_u)
        #     if iter_count == 0:
        #         r_l = r_l_0.detach()
        #         r_u = r_u_0.detach()
        #         r_l_s = r_l_0.detach()
        #         r_u_s = r_u_0.detach()
        #     else:
        #         r_l_s = torch.sigmoid(r_l).detach()
        #         r_u_s = torch.sigmoid(r_u).detach()
        #         # VAE step
        #     for count in range(num_vae_steps):  # num_vae_steps
        #         recon, mu, logvar = tavaal_vae(labeled_imgs, r_l_s)
        #         unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
        #         unlab_recon, unlab_mu, unlab_logvar = tavaal_vae(unlabeled_imgs, r_u_s)
        #         transductive_loss = vae_loss(unlabeled_imgs,
        #                                      unlab_recon, unlab_mu, unlab_logvar, beta)
        #
        #         labeled_preds = discriminator(r_l, mu)
        #         unlabeled_preds = discriminator(r_u, unlab_mu)
        #
        #         lab_real_preds = torch.ones(labeled_imgs.size(0))
        #         unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
        #
        #         with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #             lab_real_preds = lab_real_preds.cuda()
        #             unlab_real_preds = unlab_real_preds.cuda()
        #
        #         dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
        #                    bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
        #         total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
        #
        #         optim_vaal_vae.zero_grad()
        #         total_vae_loss.backward()
        #         optim_vaal_vae.step()
        #
        #         # # sample new batch if needed to train the adversarial network
        #         # if count < (num_vae_steps - 1):
        #         #     labeled_imgs, _ = next(labeled_data)
        #         #     unlabeled_imgs = next(unlabeled_data)[0]
        #         #
        #         #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #         #         labeled_imgs = labeled_imgs.cuda()
        #         #         unlabeled_imgs = unlabeled_imgs.cuda()
        #         #         labels = labels.cuda()
        #
        #     # Discriminator step
        #     for count in range(num_adv_steps):
        #         with torch.no_grad():
        #             _, mu, _ = tavaal_vae(labeled_imgs, r_l_s)
        #             _, unlab_mu, _ = tavaal_vae(unlabeled_imgs, r_u_s)
        #
        #         labeled_preds = discriminator(r_l, mu)
        #         unlabeled_preds = discriminator(r_u, unlab_mu)
        #
        #         lab_real_preds = torch.ones(labeled_imgs.size(0))
        #         unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))
        #
        #         with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #             lab_real_preds = lab_real_preds.cuda()
        #             unlab_fake_preds = unlab_fake_preds.cuda()
        #
        #         dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
        #                    bce_loss(unlabeled_preds[:, 0], unlab_fake_preds)
        #
        #         optim_discriminator.zero_grad()
        #         dsc_loss.backward()
        #         optim_discriminator.step()
        #
        #         # # sample new batch if needed to train the adversarial network
        #         # if count < (num_adv_steps - 1):
        #         #     labeled_imgs, _ = next(labeled_data)
        #         #     unlabeled_imgs = next(unlabeled_data)[0]
        #         #
        #         #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        #         #         labeled_imgs = labeled_imgs.cuda()
        #         #         unlabeled_imgs = unlabeled_imgs.cuda()
        #         #         labels = labels.cuda()
        #         if iter_count % 10 == 0:
        #             print("TA-VAAL iteration: " + str(iter_count) + " vae_loss: " + str(
        #                 total_vae_loss.item()) + " dsc_loss: " + str(dsc_loss.item()))
        #
        # all_preds, all_indices = [], []
        # for batch_x, _, batch_p in unlabeled_loader:
        #     batch_x = batch_x.float().to(device)
        #     with torch.no_grad():
        #         z_mean, _ = vae.encode(batch_x)
        #         _, features = task_model(z_mean)
        #         r = ranker(features)
        #         _, mu, _ = tavaal_vae(batch_x, torch.sigmoid(r))
        #         preds = discriminator(r, mu)
        #
        #     preds = preds.cpu().data
        #     all_preds.extend(preds)
        #     all_indices.extend(indices)
        #
        # all_preds = torch.stack(all_preds)
        # all_preds = all_preds.view(-1)
        # # need to multiply by -1 to be able to use torch.topk
        # all_preds *= -1
        # # select the points which the discriminator things are the most likely to be unlabeled
        # _, arg = torch.sort(all_preds)

        arg = np.random.randint(4000, size=4000)
        # Update the labeled dataset and the unlabeled dataset, respectively
        new_list = list(torch.tensor(unlabeled_set)[arg][:ADDENNUM].numpy())
        # print(len(new_list), min(new_list), max(new_list))
        labeled_set += list(torch.tensor(unlabeled_set)[arg][-ADDENNUM:].numpy())
        listd = list(torch.tensor(unlabeled_set)[arg][:-ADDENNUM].numpy())
        unlabeled_set = listd + unlabeled_set
        print(len(labeled_set), min(labeled_set), max(labeled_set))
        # Create a new dataloader for the updated labeled dataset
        train_loader = DataLoader(data_train, batch_size=BATCH,
                                          sampler=SubsetRandomSampler(labeled_set),
                                          pin_memory=True)



if __name__ == '__main__':
    main()
