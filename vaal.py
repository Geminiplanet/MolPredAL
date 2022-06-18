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
from models import Predictor, Discriminator, AE


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


def symbol_vec(batch_x, recon_x):
    indices1 = torch.max(batch_x, dim=2)[1]
    indices2 = torch.max(recon_x, dim=2)[1]
    for i in range(32):
        string1 = ""
        string2 = ""
        for j in range(MAX_QM9_LEN):
            string1 += QM9_CHAR_LIST[indices1[i, j]]
            string2 += QM9_CHAR_LIST[indices2[i, j]]
        print(f'x and recon_x:\n{string1} \n{string2}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    method = 'VAAL'
    results = open(f'results_{method}_QM9_{CYCLES}CYCLES.txt', 'w')
    for task_num in range(3, 5):
        # load dataset
        data_train, data_test, data_unlabeled = load_qm9_dataset('data/qm9.csv', task_num)
        indices = list(range(data_train.len))
        random.shuffle(indices)
        labeled_set = indices[:ADDENNUM]
        unlabeled_set = [x for x in indices if x not in labeled_set]
        train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        # train_loader = DataLoader(data_train, batch_size=BATCH, pin_memory=True)
        for cycle in range(CYCLES):
            # train and test task model
            print('>> Train vae and task model')
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            task_ae = AE(seq_len=MAX_QM9_LEN, fea_num=len(QM9_CHAR_LIST), hidden_dim=LATENT_DIM, layers=1)
            optim_ae = optim.Adam(task_ae.parameters(), lr=LR)
            criterion_ae = nn.CrossEntropyLoss()

            task_pred = Predictor(hidden_dim=LATENT_DIM, prop_num=1).to(device)
            optim_pred = optim.Adam(task_pred.parameters(), lr=LR)
            criterion_pred = nn.L1Loss()

            for epoch in range(121):
                task_ae.train()
                task_pred.train()
                losses = []
                for data in train_loader:
                    batch_x, batch_l, batch_p = data
                    batch_x = batch_x.to(device)
                    batch_l = batch_l.to(device)
                    batch_p = batch_p.to(device)

                    noise = torch.normal(mean=torch.zeros(BATCH, LATENT_DIM), std=0.2 * np.power(0.99, epoch) + 0.02)
                    output, z = task_ae(batch_x, batch_l, noise)
                    ae_loss = criterion_ae(output.reshape(-1, len(QM9_CHAR_LIST)), batch_x.reshape(-1))

                    # z = task_ae.Enc(batch_x, batch_l)
                    pred_p = task_pred(z)
                    target_loss = criterion_pred(pred_p, batch_p)

                    loss = target_loss + ae_loss

                    optim_ae.zero_grad()
                    optim_pred.zero_grad()
                    loss.backward(retain_graph=True)
                    optim_ae.step()
                    optim_pred.step()

                    losses.append(loss.item())

                train_loss = np.array(losses).mean()
                if epoch % 10 == 0:
                    # symbol_vec(batch_x, recon_x)
                    print(f'epoch {epoch}: train loss is {train_loss: .5f}')

            # test task model
            print(" >> Test Model")
            task_ae.eval()
            task_pred.eval()
            outputs = []
            labels = []
            with torch.no_grad():
                for data in test_loader:
                    batch_x, batch_l, batch_p = data
                    batch_x = batch_x.to(device)
                    batch_l = batch_l.to(device)
                    z = task_ae.Enc(batch_x, batch_l)
                    output = task_pred(z)
                    outputs.append(output.cpu())
                    labels.append(batch_p.cpu())
            test_loss = criterion_pred(torch.cat(outputs).view(-1), torch.cat(labels)).item()
            print(
                f'Cycle {cycle + 1}/{CYCLES} || labeled data size {len(labeled_set)}, test loss(MAE) = {test_loss: .5f}')
            np.array([method, QM9_TASKS[task_num], cycle + 1, CYCLES, len(labeled_set), test_loss]).tofile(results,
                                                                                                           sep=" ")
            results.write("\n")

            if cycle == CYCLES - 1:
                print('Finished.')
                break
            if method == 'VAAL':
                # AL to select data
                # Get the indices of the unlabeled samples to train on next cycle
                unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(subset), pin_memory=True)
                labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                            sampler=SubsetSequentialSampler(labeled_set),
                                            pin_memory=True)

                # train VAAL
                vaal_ae = AE(MAX_QM9_LEN, len(QM9_CHAR_LIST), LATENT_DIM, 1).to(device)
                discriminator = Discriminator(LATENT_DIM).to(device)
                optim_vaal_ae = optim.Adam(vaal_ae.parameters(), lr=LR)
                optim_discriminator = optim.Adam(discriminator.parameters(), lr=LR)
                vaal_ae.train()
                discriminator.train()

                adversary_param = 1
                beta = 1
                num_adv_steps = 1
                num_vae_steps = 1

                bce_loss = nn.BCELoss()

                labeled_data = read_data(labeled_loader)
                unlabeled_data = read_data(unlabeled_loader, labels=False)

                train_iterations = (ADDENNUM * cycle + len(subset)) * 30 // BATCH

                for iter_count in range(train_iterations):
                    labeled_x, labeled_l, labeled_p = next(labeled_data)
                    unlabeled_x, unlabeled_l = next(unlabeled_data)

                    labeled_x = labeled_x.to(device)
                    labeled_l = labeled_l.to(device)

                    unlabeled_x = unlabeled_x.to(device)
                    unlabeled_l = unlabeled_l.to(device)

                    # VAE step
                    for count in range(num_vae_steps):  # num_vae_steps
                        noise = torch.normal(mean=torch.zeros(BATCH, LATENT_DIM),
                                             std=0.2 * np.power(0.99, iter_count) + 0.02)
                        recon, z = vaal_ae(labeled_x, labeled_l, noise)
                        unsup_loss = criterion_ae(recon.reshape(-1, len(QM9_CHAR_LIST)), labeled_x.reshape(-1))
                        unlab_recon, unlab_z = vaal_ae(unlabeled_x, unlabeled_l, noise)
                        transductive_loss = criterion_ae(unlab_recon.reshape(-1, len(QM9_CHAR_LIST)),
                                                         unlabeled_x.reshape(-1))

                        labeled_preds = discriminator(z)
                        unlabeled_preds = discriminator(unlab_z)

                        lab_real_preds = torch.ones(labeled_x.size(0))
                        unlab_real_preds = torch.ones(unlabeled_x.size(0))

                        lab_real_preds = lab_real_preds.to(device)
                        unlab_real_preds = unlab_real_preds.to(device)

                        dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                                   bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
                        total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
                        # print(f"unsup_loss: {unsup_loss:.5f} transductive_loss:{transductive_loss:.5f} dsc_loss:{dsc_loss:.5f}")
                        optim_vaal_ae.zero_grad()
                        total_vae_loss.backward()
                        optim_vaal_ae.step()

                        # # sample new batch if needed to train the adversarial network
                        # if count < (num_vae_steps - 1):
                        #     labeled_imgs, _ = next(labeled_data)
                        #     unlabeled_imgs = next(unlabeled_data)[0]
                        #
                        #     with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                        #         labeled_imgs = labeled_imgs.cuda()
                        #     unlabeled_imgs = unlabeled_imgs.cuda()
                        #     labels = labels.cuda()

                    # Discriminator step
                    for count in range(num_adv_steps):
                        with torch.no_grad():
                            z = vaal_ae.Enc(labeled_x, labeled_l)
                            unlab_z = vaal_ae.Enc(unlabeled_x, unlabeled_l)

                        labeled_preds = discriminator(z)
                        unlabeled_preds = discriminator(unlab_z)

                        lab_real_preds = torch.ones(labeled_x.size(0))
                        unlab_fake_preds = torch.zeros(unlabeled_x.size(0))

                        lab_real_preds = lab_real_preds.to(device)
                        unlab_fake_preds = unlab_fake_preds.to(device)

                        dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                                   bce_loss(unlabeled_preds[:, 0], unlab_fake_preds)

                        # dsc_loss = -torch.mean(torch.log(labeled_preds) + torch.log(1 - unlabeled_preds))
                        optim_discriminator.zero_grad()
                        dsc_loss.backward()
                        optim_discriminator.step()

                    if (iter_count % 10 == 0 and iter_count < 100) or iter_count % 100 == 0:
                        print(f"VAAL iteration: {iter_count} vae_loss: {total_vae_loss: .5f} dsc_loss: {dsc_loss: .5f}")

                all_preds, all_indices = [], []
                for batch_x, batch_l, _ in unlabeled_loader:
                    batch_x = batch_x.to(device)
                    batch_l = batch_l.to(device)
                    with torch.no_grad():
                        z = vaal_ae.Enc(batch_x, batch_l)
                        preds = discriminator(z)

                    preds = preds.cpu().data
                    all_preds.extend(preds)
                    # all_indices.extend(indices)
                all_preds = torch.stack(all_preds)
                all_preds = all_preds.view(-1)
                # need to multiply by -1 to be able to use torch.topk
                all_preds *= -1
                # select the points which the discriminator things are the most likely to be unlabeled
                _, arg = torch.sort(all_preds)
            elif method == 'RANDOM':
                arg = np.random.randint(len(subset), size=len(subset))
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENNUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENNUM].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), len(unlabeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            train_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)


if __name__ == '__main__':
    main()
