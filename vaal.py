import random

import numpy as np
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from config import *
from data.sampler import SubsetSequentialSampler
from dataset import load_qm9_dataset
from models import MolecularVAE, Predictor, Discriminator


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
            for data, label in dataloader:
                yield data.float(), label
    else:
        while True:
            for data, _, in dataloader:
                yield data.float()


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
    method = 'RANDOM'
    results = open(f'results_{method}_QM9_{CYCLES}CYCLES.txt', 'w')
    for task_num in range(8):
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
        # train_loader = DataLoader(data_train, batch_size=BATCH, pin_memory=True)
        for cycle in range(CYCLES):
            # train and test task model
            print('>> Train vae and task model')
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]
            task_vae = MolecularVAE().to(device)
            optim_vae = optim.Adam(task_vae.parameters(), lr=LR)
            criterion = torch.nn.L1Loss()
            task_model = Predictor().to(device)
            optim_task = optim.Adam(task_model.parameters(), lr=LR)
            for epoch in range(120):
                task_model.train()
                task_vae.train()
                losses = []
                for data in train_loader:
                    optim_task.zero_grad()
                    optim_vae.zero_grad()
                    batch_x, batch_p = data
                    batch_x = batch_x.float().to(device)
                    batch_p = batch_p.to(device)
                    recon_x, z_mean, z_logvar = task_vae(batch_x)
                    output = task_model(z_mean)
                    target_loss = criterion(output.view(-1), batch_p)
                    loss = target_loss + vae_loss(batch_x, recon_x, z_mean, z_logvar, BETA)
                    # print(target_loss, vae_loss(batch_x, recon_x, z_mean, z_logvar, BETA))
                    loss.backward()
                    optim_task.step()
                    optim_vae.step()
                    losses.append(loss.item())
                train_loss = np.array(losses).mean()
                if epoch % 10 == 0:
                    # symbol_vec(batch_x, recon_x)
                    print(f'epoch {epoch}: train loss is {train_loss: .5f}')

            # test task model
            print(" >> Test Model")
            task_vae.eval()
            task_model.eval()
            outputs = []
            labels = []
            with torch.no_grad():
                for data in test_loader:
                    batch_x, batch_p = data
                    batch_x = batch_x.float().to(device)
                    z_mean, _ = task_vae.encode(batch_x)
                    output = task_model(z_mean)
                    outputs.append(output.cpu())
                    labels.append(batch_p.cpu())
            test_loss = criterion(torch.cat(outputs).view(-1), torch.cat(labels)).item()
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
                labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set),
                                            pin_memory=True)

                # train VAAL
                vaal_vae = MolecularVAE().to(device)
                discriminator = Discriminator(LATENT_DIM).to(device)
                optim_vaal_vae = optim.Adam(vaal_vae.parameters(), lr=LR)
                optim_discriminator = optim.Adam(discriminator.parameters(), lr=LR)
                task_model.to(device)
                task_vae.to(device)
                vaal_vae.train()
                discriminator.train()

                adversary_param = 1
                beta = 1
                num_adv_steps = 1
                num_vae_steps = 1

                bce_loss = nn.BCELoss()

                labeled_data = read_data(labeled_loader)
                unlabeled_data = read_data(unlabeled_loader)

                train_iterations = (ADDENNUM * cycle + len(subset)) * 10 // BATCH

                for iter_count in range(train_iterations):
                    labeled_imgs, labels = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)

                    # VAE step
                    for count in range(num_vae_steps):  # num_vae_steps

                        recon, mu, logvar = vaal_vae(labeled_imgs)
                        unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
                        unlab_recon, unlab_mu, unlab_logvar = vaal_vae(unlabeled_imgs)
                        transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)

                        labeled_preds = discriminator(mu)
                        unlabeled_preds = discriminator(unlab_mu)

                        lab_real_preds = torch.ones(labeled_imgs.size(0))
                        unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                        lab_real_preds = lab_real_preds.to(device)
                        unlab_real_preds = unlab_real_preds.to(device)

                        # lab_real_preds = torch.ones(labeled_imgs.size(0))
                        # unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))
                        #
                        # lab_real_preds = lab_real_preds.to(device)
                        # unlab_fake_preds = unlab_fake_preds.to(device)

                        dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                                   bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
                        total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
                        # print(f"unsup_loss: {unsup_loss:.5f} transductive_loss:{transductive_loss:.5f}")

                        optim_vaal_vae.zero_grad()
                        total_vae_loss.backward()
                        optim_vaal_vae.step()
                        # # sample new batch if needed to train the adversarial network
                        if count < (num_vae_steps - 1):
                            labeled_imgs, _ = next(labeled_data)
                        unlabeled_imgs = next(unlabeled_data)[0]

                        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                            labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

                    # Discriminator step
                    for count in range(num_adv_steps):
                        with torch.no_grad():
                            _, mu, _ = vaal_vae(labeled_imgs)
                            _, unlab_mu, _ = vaal_vae(unlabeled_imgs)

                        labeled_preds = discriminator(mu)
                        unlabeled_preds = discriminator(unlab_mu)

                        lab_real_preds = torch.ones(labeled_imgs.size(0))
                        unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

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
                for batch_x, _ in unlabeled_loader:
                    batch_x = batch_x.float().to(device)
                    with torch.no_grad():
                        _, mu, _ = vaal_vae(batch_x)
                        preds = discriminator(mu)

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
