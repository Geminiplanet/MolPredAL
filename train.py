from torch.utils.data import DataLoader

from config import *
from dataset import load_qm9_dataset
from models import Encoder, Decoder


def train():
    train_data, _, _ = load_qm9_dataset('data/qm9.csv', 3)
    encoder = Encoder(MAX_QM9_LEN, len(QM9_CHAR_LIST), LATENT_DIM, layers=1)
    decoder = Decoder(MAX_QM9_LEN, len(QM9_CHAR_LIST), LATENT_DIM, layers=1)
    train_loader = DataLoader(train_data, batch_size=BATCH)
    for data in train_loader:
        X, L, _ = data
        Z = encoder(X, L)
        decoder(Z, X)


if __name__ == '__main__':
    train()