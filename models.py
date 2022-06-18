import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


def vec_to_char(out_num):
    stri = ""
    for cha in out_num:
        stri += QM9_CHAR_LIST[cha]
    return stri


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[10, 5], num_channels=[15, 15], interm_dim=15):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool1d(feature_sizes[0])
        self.GAP2 = nn.AvgPool1d(feature_sizes[1])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)

        self.linear = nn.Linear(2 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out = self.linear(torch.cat((out1, out2), 1))
        return out


class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(42, 9, kernel_size=5)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=5)
        self.conv_3 = nn.Conv1d(9, 11, kernel_size=5)
        # self.conv_1 = nn.Conv1d(200, 18, kernel_size=18)
        # self.conv_2 = nn.Conv1d(18, 18, kernel_size=18)
        # self.conv_3 = nn.Conv1d(18, 20, kernel_size=22)

        self.linear_0 = nn.Linear(110, 250)
        # self.linear_0 = nn.Linear(400, 435)
        self.linear_1 = nn.Linear(250, LATENT_DIM)
        self.linear_2 = nn.Linear(250, LATENT_DIM)

        # vaal: nn.Linear(LATENT_DIM, 300); ta-vaal: nn.Linear(LATENT_DIM + 1, 300)
        self.linear_3 = nn.Linear(LATENT_DIM + 1, 300)
        self.gru = nn.GRU(300, 500, 3, batch_first=True)
        self.linear_4 = nn.Linear(500, len(QM9_CHAR_LIST))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        # x = self.relu(self.conv(x))
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, MAX_QM9_LEN, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    # ta-vaal
    def forward(self, x, r):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        z = torch.cat([z, r], 1)
        return self.decode(z), z_mean, z_logvar

    # def forward(self, x):
    #     z_mean, z_logvar = self.encode(x)
    #     z = self.sampling(z_mean, z_logvar)
    #     return self.decode(z), z_mean, z_logvar


# ta-vaal predictor

class Predictor(nn.Module):
    def __init__(self, hidden_dim, prop_num):
        super(Predictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.prop_num = prop_num

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)

        self.fc3 = nn.Linear(self.hidden_dim, self.prop_num)
        nn.init.xavier_normal_(self.fc3.weight.data)
        nn.init.normal_(self.fc3.bias.data)

        self.relu = nn.ReLU()

    def forward(self, z):
        out1 = self.relu(self.fc1(z))
        out2 = self.relu(self.fc2(out1))
        out = self.fc3(out2)

        return out.view(-1)  # , [out1, out2]


# vaal predictor
# class Predictor(nn.Module):
#     def __init__(self):
#         super(Predictor, self).__init__()
#
#         self.fc1 = nn.Linear(LATENT_DIM, 1)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, z_mean):
#         return self.sigmoid(self.fc1(z_mean))


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=LATENT_DIM):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # nn.Linear(z_dim + 1, 500),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    # def forward(self, r, z):
    #     z = torch.cat([z, r], 1)
    #     return self.net(z)
    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Encoder(nn.Module):

    def __init__(self, seq_len, fea_num, hidden_dim, layers):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.fea_num = fea_num
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.embedd = nn.Embedding(fea_num, fea_num)
        self.encoder_rnn = nn.LSTM(input_size=self.fea_num, hidden_size=self.hidden_dim, num_layers=self.layers,
                                   bias=True, batch_first=True, bidirectional=False)

        for param in self.encoder_rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, X, L):
        batch_size = X.shape[0]
        device = X.device
        enc_h0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)
        enc_c0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)

        X = self.embedd(X)
        out, (encoder_hn, encoder_cn) = self.encoder_rnn(X, (enc_h0, enc_c0))
        last_step_index_list = (L - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        Z = out.gather(1, last_step_index_list).squeeze()
        # Z=torch.sigmoid(Z)
        Z = F.normalize(Z, p=2, dim=1)

        return Z


class Decoder(nn.Module):

    def __init__(self, seq_len, fea_num, hidden_dim, layers):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.fea_num = fea_num
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.embedd = nn.Embedding(self.fea_num, self.fea_num)
        self.decoder_rnn = nn.LSTM(input_size=self.fea_num + self.hidden_dim, hidden_size=self.hidden_dim,
                                   num_layers=self.layers, bias=True, batch_first=True, bidirectional=False)

        for param in self.decoder_rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        self.decoder_fc1 = nn.Linear(self.hidden_dim, self.fea_num)
        nn.init.xavier_normal_(self.decoder_fc1.weight.data)
        nn.init.normal_(self.decoder_fc1.bias.data)

    def forward(self, Z, X):

        batch_size = Z.shape[0]
        device = Z.device
        dec_h0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)

        X = self.embedd(X)
        Zm = Z.view(-1, 1, self.hidden_dim).expand(-1, self.seq_len, self.hidden_dim)
        ZX = torch.cat((Zm, X), 2)
        #        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(X0,(Z.view(1,-1,self.hidden_dim),dec_c0))
        dec_out, (decoder_hn, decoder_cn) = self.decoder_rnn(ZX, (dec_h0, dec_c0))
        dec = self.decoder_fc1(dec_out)
        return dec

    def decoding(self, Z):
        batch_size = Z.shape[0]
        device = Z.device
        dec_h0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.layers * 1, batch_size, self.hidden_dim).to(device)

        seq = torch.zeros([batch_size, 1], dtype=torch.long).to(device)

        Y = seq
        Zm = Z.view(-1, 1, self.hidden_dim).expand(-1, 1, self.hidden_dim)

        decoder_hn = dec_h0
        decoder_cn = dec_c0
        #        seq2=Xdata_onehot
        for i in range(self.seq_len):
            dec_h0 = decoder_hn
            dec_c0 = decoder_cn

            X = self.embedd(Y)
            ZX = torch.cat((Zm, X), 2)
            dec_out, (decoder_hn, decoder_cn) = self.decoder_rnn(ZX, (dec_h0, dec_c0))
            dec = self.decoder_fc1(dec_out)
            Y = torch.argmax(dec, dim=2)
            #            Xdata_onehot=torch.zeros([batch_size,self.Nfea],dtype=torch.float32).to(device)
            #            Xdata_onehot=Xdata_onehot.scatter_(1,Y,1).view(-1,1,self.Nfea)
            seq = torch.cat((seq, Y), dim=1)
        #            seq2=torch.cat((seq2,dec),dim=1)

        return seq[:, 1:]  # , seq2[:,1:]


class AE(nn.Module):

    def __init__(self, seq_len, fea_num, hidden_dim, layers):
        super(AE, self).__init__()

        self.seq_len = seq_len
        self.fea_num = fea_num
        # self.prop_num = prop_num
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.Enc = Encoder(seq_len, fea_num, hidden_dim, layers)
        self.Dec = Decoder(seq_len, fea_num, hidden_dim, layers)
        # self.Pred = Predictor(hidden_dim, prop_num)

    def forward(self, X, L, noise):
        Z = self.Enc(X, L)
        Zn = Z + noise
        decoded = self.Dec(Zn, X)

        return decoded, Z
