import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import optim, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from config import *
from dataset import load_qm9_dataset
from models import AE, Predictor

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for task_num in range(12):
        data_train, data_test, data_all = load_qm9_dataset('data/qm9.csv', task_num)
        indices = list(range(data_train.len))
        random.shuffle(indices)
        labeled_set = indices[:SUBSET]
        data_loader = DataLoader(data_all, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                                 pin_memory=True)

        print('>> Train vae and task model')

        task_ae = AE(seq_len=MAX_QM9_LEN, fea_num=len(QM9_CHAR_LIST), hidden_dim=LATENT_DIM, layers=1).to(device)
        optim_ae = optim.Adam(task_ae.parameters(), lr=LR)
        criterion_ae = nn.CrossEntropyLoss()

        task_pred = Predictor(hidden_dim=LATENT_DIM, prop_num=1).to(device)
        optim_pred = optim.Adam(task_pred.parameters(), lr=LR)
        criterion_pred = nn.L1Loss()

        for epoch in range(101):
            task_ae.train()
            task_pred.train()
            losses = []
            for data in tqdm(data_loader, leave=False):
                batch_x, batch_l, batch_p = data
                batch_x = batch_x.to(device)
                batch_l = batch_l.to(device)
                batch_p = batch_p.to(device)

                noise = torch.normal(mean=torch.zeros(BATCH, LATENT_DIM), std=0.2 * np.power(0.99, epoch) + 0.02)
                noise = noise.to(device)
                output, z = task_ae(batch_x, batch_l, noise)
                ae_loss = criterion_ae(output.reshape(-1, len(QM9_CHAR_LIST)), batch_x.reshape(-1))

                # z = task_ae.Enc(batch_x, batch_l)
                pred_p = task_pred(z)
                target_loss = criterion_pred(pred_p, batch_p)

                loss = target_loss + ae_loss

                optim_ae.zero_grad()
                optim_pred.zero_grad()
                loss.backward()
                optim_ae.step()
                optim_pred.step()

                losses.append(loss.item())

            train_loss = np.array(losses).mean()
            if epoch % 10 == 0:
                print(f'epoch {epoch}: train loss is {train_loss: .5f}')

        x, l, y = data_all[labeled_set]
        x = x.to(device)
        l = l.to(device)
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = plt.cm.get_cmap('RdYlBu_r')
        z = task_ae.Enc(x, l)

        z = z.detach().numpy()
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z)

        sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], marker='.', c=y, cmap=cm)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'property:{QM9_TASKS[task_num]}')
        plt.colorbar(sc, orientation='vertical')

        plt.savefig(f'img/{QM9_TASKS[task_num]}.jpg')
        plt.show()
