import random

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from config import *
from data.sampler import SubsetSequentialSampler
from dataset import load_qm9_dataset
from models import MolecularVAE, Predictor, LossNet, Discriminator, AE


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
    recon_loss = F.binary_cross_entropy(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss * beta
    return recon_loss + kl_loss


# def vae_loss(x, recon, mu, logvar, beta):
#     mse_loss = nn.MSELoss()
#     MSE = mse_loss(recon, x)
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     KLD = KLD * beta
#     return MSE + KLD


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for data, length, prop in dataloader:
                yield data, length, prop
    else:
        while True:
            for data, length, _ in dataloader:
                yield data, length


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    method = 'TA-VAAL'
    # results = open(f'results_{method}_TOX21_{CYCLES}CYCLES.txt', 'w')
    results = open(f'results_{method}_QM9_{CYCLES}CYCLES.txt', 'w')
    for task_num in range(6):
        # load dataset
        data_train, data_test, _ = load_qm9_dataset('data/qm9.csv', task_num)
        data_unlabeled = data_train
        indices = list(range(data_train.len))
        random.shuffle(indices)
        labeled_set = indices[:ADDENNUM]
        unlabeled_set = [x for x in indices if x not in labeled_set]
        train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        # train and test task model
        for cycle in range(CYCLES):
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]
            print('>> Train vae and task model')

            # train vae
            # scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optim_vae, 'min', factor=0.8, patience=3,
            #                                                      min_lr=0.0001)
            # for epoch in range(1, 121):
            #     task_vae.train()
            #     epoch_loss = 0
            #     for data in train_loader:
            #         batch_x, batch_p = data
            #         batch_x = batch_x.float().to(device)
            #         optim_vae.zero_grad()
            #         # ta-vaal needs r(rank)
            #         r = batch_p.float().unsqueeze(1).to(device)
            #         recon_x, z_mean, z_logvar = task_vae(batch_x, r)
            #         # recon_x, z_mean, z_logvar = task_vae(batch_x)
            #
            #         loss = vae_loss(batch_x, recon_x, z_mean, z_logvar, BETA)
            #         loss.backward()
            #         epoch_loss += loss
            #         optim_vae.step()
            #
            #     if epoch % 20 == 0:
            #         print(f'epoch {epoch}, vae loss is {epoch_loss}')
            #         symbol_vec(batch_x, recon_x)
            # scheduler_vae.step(epoch_loss)

            # train task model and loss module
            task_ae = AE(seq_len=MAX_QM9_LEN, fea_num=len(QM9_CHAR_LIST), hidden_dim=LATENT_DIM, layers=1).to(device)
            optim_ae = optim.Adam(task_ae.parameters(), lr=LR)
            criterion_ae = nn.CrossEntropyLoss()

            task_pred = Predictor(hidden_dim=LATENT_DIM, prop_num=1).to(device)
            optim_pred = optim.Adam(task_pred.parameters(), lr=LR)
            criterion_pred = nn.L1Loss(reduction="none")
            loss_module = LossNet().to(device)
            optim_module = optim.Adam(loss_module.parameters(), lr=LR, weight_decay=WDECAY)
            for epoch in range(121):
                task_ae.train()
                task_pred.train()
                loss_module.train()
                losses = []
                for data in train_loader:
                    batch_x, batch_l, batch_p = data
                    batch_x = batch_x.to(device)
                    batch_l = batch_l.to(device)
                    batch_p = batch_p.to(device)
                    optim_ae.zero_grad()
                    optim_pred.zero_grad()
                    optim_module.zero_grad()
                    r = batch_p.unsqueeze(1).to(device)
                    noise = torch.normal(mean=torch.zeros(BATCH, LATENT_DIM),
                                         std=0.2 * np.power(0.99, epoch) + 0.02).to(device)
                    output, z = task_ae(batch_x, batch_l, noise, r)

                    scores, features = task_pred(z)
                    target_loss = criterion_pred(scores.view(-1), batch_p)
                    pred_loss = loss_module(features)
                    pred_loss = pred_loss.view(pred_loss.shape[0])
                    m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
                    # m_backbone_loss = vae_loss(batch_x, recon_x, z_mean, z_logvar, BETA) + torch.sum(
                    #     target_loss) / target_loss.size(0)
                    t = torch.sum(target_loss) / target_loss.size(0)
                    tt = torch.mean(target_loss)
                    m_backbone_loss = torch.sum(target_loss) / target_loss.size(0) + criterion_ae(
                        output.reshape(-1, len(QM9_CHAR_LIST)), batch_x.reshape(-1))
                    loss = m_backbone_loss + WEIGHT * m_module_loss
                    loss.backward()
                    optim_ae.step()
                    optim_pred.step()
                    optim_module.step()
                    losses.append(loss.item())
                train_loss = np.array(losses).mean()
                # sched_task.step()
                # sched_module.step()
                if epoch % 10 == 0:
                    # symbol_vec(batch_x, recon_x)
                    print(
                        f'epoch {epoch}: train loss is {train_loss: .5f}')

            # test task model
            print(" >> Test Model")
            task_ae.eval()
            task_pred.eval()
            loss_module.eval()
            labels = []
            outputs = []
            cri = nn.L1Loss()
            test_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    batch_x, batch_l, batch_p = data
                    batch_x = batch_x.to(device)
                    batch_l = batch_l.to(device)
                    z = task_ae.Enc(batch_x, batch_l)
                    output, _ = task_pred(z)
                    test_loss += cri(output.cpu(), batch_p)
                    # z_mean, _ = task_vae.encode(batch_x)
                    # scores, _ = task_model(z_mean)
                    # outputs.append(scores.cpu())
                    # labels.append(batch_p.cpu())
            # test_loss = np.array(criterion_pred(torch.cat(outputs).view(-1), torch.cat(labels).view(-1))).sum()
            print(
                f'Cycle {cycle + 1}/{CYCLES} || labeled data size {len(labeled_set)}, test loss(MAE) = {test_loss: .5f}')
            np.array([method, QM9_TASKS[task_num], cycle + 1, CYCLES, len(labeled_set), test_loss]).tofile(results,
                                                                                                           sep=" ")
            results.write("\n")

            if cycle == CYCLES - 1:
                print('Finished.')
                break

            # AL to select data
            # Get the indices of the unlabeled samples to train on next cycle
            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset), pin_memory=True)
            labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set),
                                        pin_memory=True)

            # train TA-VAAL
            # tavaal_vae = MolecularVAE().to(device)
            # discriminator = Discriminator(LATENT_DIM).to(device)
            # optim_vaal_vae = optim.Adam(tavaal_vae.parameters(), lr=LR)
            # optim_discriminator = optim.Adam(discriminator.parameters(), lr=LR)
            tavaal_ae = AE(MAX_QM9_LEN, len(QM9_CHAR_LIST), LATENT_DIM, 1).to(device)
            discriminator = Discriminator(LATENT_DIM).to(device)
            optim_tavaal_ae = optim.Adam(tavaal_ae.parameters(), lr=LR)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=LR)
            tavaal_ae.train()
            discriminator.train()
            ranker = loss_module
            task_pred.eval().to(device)
            task_ae.eval().to(device)
            ranker.eval().to(device)
            tavaal_ae.train()
            discriminator.train()

            adversary_param = 1
            beta = 1
            num_adv_steps = 1
            num_vae_steps = 1

            bce_loss = nn.BCELoss()

            labeled_data = read_data(labeled_loader)
            unlabeled_data = read_data(unlabeled_loader, labels=False)

            train_iterations = int((ADDENNUM * cycle + SUBSET) * 30 / BATCH)

            for iter_count in range(train_iterations):
                labeled_x, labeled_l, labeled_p = next(labeled_data)
                unlabeled_x, unlabeled_l = next(unlabeled_data)

                labeled_x = labeled_x.to(device)
                labeled_l = labeled_l.to(device)

                unlabeled_x = unlabeled_x.to(device)
                unlabeled_l = unlabeled_l.to(device)
                # labels = labels.cuda()
                if iter_count == 0:
                    r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_x.shape[0], 1))).type(
                        torch.FloatTensor).cuda()
                    r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_x.shape[0], 1))).type(
                        torch.FloatTensor).cuda()
                else:
                    with torch.no_grad():
                        labeled_input, _ = task_ae.encode(labeled_x)
                        _, features_l = task_pred(labeled_input)
                        unlabeled_input, _ = task_ae.encode(unlabeled_x)
                        _, feature_u = task_pred(unlabeled_input)
                        r_l = ranker(features_l)
                        r_u = ranker(feature_u)
                if iter_count == 0:
                    r_l = r_l_0.detach()
                    r_u = r_u_0.detach()
                    r_l_s = r_l_0.detach()
                    r_u_s = r_u_0.detach()
                else:
                    r_l_s = torch.sigmoid(r_l).detach()
                    r_u_s = torch.sigmoid(r_u).detach()

                # VAE step
                for count in range(num_vae_steps):  # num_vae_steps
                    optim_tavaal_ae.zero_grad()
                    noise = torch.normal(mean=torch.zeros(BATCH, LATENT_DIM),
                                         std=0.2 * np.power(0.99, iter_count) + 0.02).to(device)

                    recon, z = tavaal_ae(labeled_x, labeled_l, noise, r_l_s)
                    unsup_loss = criterion_ae(recon.reshape(-1, len(QM9_CHAR_LIST)), labeled_x.reshape(-1))
                    unlab_recon, unlab_z = tavaal_ae(unlabeled_x, unlabeled_l, noise, r_u_s)
                    transductive_loss = criterion_ae(unlab_recon.reshape(-1, len(QM9_CHAR_LIST)),
                                                         unlabeled_x.reshape(-1))

                    labeled_preds = discriminator(r_l, z)
                    unlabeled_preds = discriminator(r_u, unlab_z)

                    lab_real_preds = torch.ones(labeled_x.size(0))
                    unlab_real_preds = torch.ones(unlabeled_x.size(0))

                    lab_real_preds = lab_real_preds.to(device)
                    unlab_real_preds = unlab_real_preds.to(device)

                    dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                               bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
                    total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss

                    total_vae_loss.backward()
                    optim_tavaal_ae.step()

                    # # sample new batch if needed to train the adversarial network
                    # if count < (num_vae_steps - 1):
                    #     labeled_imgs, _ = next(labeled_data)
                    #     unlabeled_imgs = next(unlabeled_data)[0]
                    #
                    #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    #         labeled_imgs = labeled_imgs.cuda()
                    #         unlabeled_imgs = unlabeled_imgs.cuda()
                    #         labels = labels.cuda()

                # Discriminator step
                for count in range(num_adv_steps):
                    optim_discriminator.zero_grad()

                    with torch.no_grad():
                        z = tavaal_ae.Enc(labeled_x, labeled_l)
                        unlab_z = tavaal_ae.Enc(unlabeled_x, unlabeled_l)

                    labeled_preds = discriminator(r_l, z)
                    unlabeled_preds = discriminator(r_u, unlab_z)

                    lab_real_preds = torch.ones(labeled_x.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_x.size(0))

                    lab_real_preds = lab_real_preds.to(device)
                    unlab_fake_preds = unlab_fake_preds.to(device)

                    dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                               bce_loss(unlabeled_preds[:, 0], unlab_fake_preds)

                    dsc_loss.backward()
                    optim_discriminator.step()

                    # # sample new batch if needed to train the adversarial network
                    # if count < (num_adv_steps - 1):
                    #     labeled_imgs, _ = next(labeled_data)
                    #     unlabeled_imgs = next(unlabeled_data)[0]
                    #
                    #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    #         labeled_imgs = labeled_imgs.cuda()
                    #         unlabeled_imgs = unlabeled_imgs.cuda()
                    #         labels = labels.cuda()
                    if (iter_count % 10 == 0 and iter_count < 100) or iter_count % 100 == 0:
                        print("TA-VAAL iteration: " + str(iter_count) + " vae_loss: " + str(
                            total_vae_loss.item()) + " dsc_loss: " + str(dsc_loss.item()))

            all_preds, all_indices = [], []
            for batch_x, batch_l, _ in unlabeled_loader:
                batch_x = batch_x.to(device)
                batch_l = batch_l.to(device)
                with torch.no_grad():
                    z = task_ae.Enc(batch_x, batch_l)
                    _, features = task_pred(z)
                    r = ranker(features)
                    mu = tavaal_ae.Enc(batch_x, batch_l)
                    preds = discriminator(r, mu)

                preds = preds.cpu().data
                all_preds.extend(preds)
                # all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk
            all_preds *= -1
            # select the points which the discriminator things are the most likely to be unlabeled
            _, arg = torch.sort(all_preds)

            # random
            # arg = np.random.randint(len(subset), size=len(subset))
            # Update the labeled dataset and the unlabeled dataset, respectively
            # new_list = list(torch.tensor(unlabeled_set)[arg][:ADDENNUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-ADDENNUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENNUM].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), len(unlabeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)
    results.close()


if __name__ == '__main__':
    main()
