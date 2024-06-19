from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import utils
from torch.nn import init, TransformerEncoder, TransformerEncoderLayer
import math


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.num_features = 5000
        self.n_diagnosis_codes = 3
        self.emb_size = 8
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size, padding_idx=-1, max_norm=1.0)
        self.hidden_size = 16
        self.num_classes = 2
        self.num_new_features = 256
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(self.num_new_features),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.num_new_features * self.emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def embedding(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.reshape(x.size(0), -1)
        # x = self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers=1, embedding_dim=8, hidden_dim=32, num_heads=4, dropout=0):
        super(Transformer, self).__init__()
        num_classes = 2
        self.n_diagnosis_codes = 3
        self.emb = nn.Embedding(self.n_diagnosis_codes, embedding_dim, padding_idx=-1, max_norm=1.0)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))
        self.shorten = nn.Linear(5000, 128)
        # self.projection = nn.Sequential(
        #     nn.Linear(embedding_dim, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )

    def forward(self, x):
        # x.shape = (batch_size, seq_len)
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.shorten(x)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x)
        output = output.transpose(0, 1)  # output.shape = (batch_size, seq_len, embedding_dim)
        output = torch.sum(output, dim=1)
        # energy = self.projection(output)
        # weights = F.softmax(energy.squeeze(-1), dim=1)
        # # (B, L, H) * (B, L, 1) -> (B, H)
        # output = (output * weights.unsqueeze(-1)).sum(dim=1)
        output = self.fc(output)
        return output

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False


class Transformer_E(nn.Module):
    def __init__(self, num_layers=1, embedding_dim=8, hidden_dim=16, num_heads=4, dropout=0.1):
        super(Transformer_E, self).__init__()
        self.n_diagnosis_codes = 3
        self.emb = nn.Embedding(self.n_diagnosis_codes, embedding_dim, padding_idx=-1, max_norm=1.0)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))
        self.shorten = nn.Linear(5000, 128)
        # self.projection = nn.Sequential(
        #     nn.Linear(embedding_dim, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )
        self.hidden_size = embedding_dim

    def forward(self, x):
        # x.shape = (batch_size, seq_len)
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.shorten(x)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x)
        output = output.transpose(0, 1)  # output.shape = (batch_size, seq_len, embedding_dim)
        output = torch.sum(output, dim=1)
        # energy = self.projection(output)
        # weights = F.softmax(energy.squeeze(-1), dim=1)
        # # (B, L, H) * (B, L, 1) -> (B, H)
        # output = (output * weights.unsqueeze(-1)).sum(dim=1)
        return output


class Transformer_Dc(nn.Module):
    def __init__(self, num_layers=3, embedding_dim=8, hidden_dim=16, num_heads=8, dropout=0.1):
        super(Transformer_Dc, self).__init__()
        num_classes = 2
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, output):
        output = self.fc(output)
        return output


class MLP_nobn(nn.Module):
    def __init__(self):
        super(MLP_nobn, self).__init__()
        self.num_features = 5000
        self.n_diagnosis_codes = 3
        self.emb_size = 8
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size, padding_idx=-1, max_norm=1.0)
        self.hidden_size = 16
        self.num_classes = 2
        self.num_new_features = 256
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_new_features * self.emb_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False


class MLP_E(nn.Module):
    def __init__(self):
        super(MLP_E, self).__init__()
        self.num_features = 5000
        self.n_diagnosis_codes = 3
        self.emb_size = 8
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size, padding_idx=-1, max_norm=1.0)
        self.hidden_size = 16
        self.num_classes = 2
        self.num_new_features = 256
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.batchnorm = nn.Sequential(
            nn.BatchNorm1d(self.num_new_features),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.num_new_features * self.emb_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class MLP_Dc(nn.Module):
    def __init__(self):
        super(MLP_Dc, self).__init__()
        self.hidden_size = 16
        self.num_classes = 2
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class SNPDpedec(nn.Module):

  def __init__(self, encoder, num_features, num_classes=0):
    super(SNPDpedec, self).__init__()
    self.encoder = encoder
    self.num_features = num_features
    self.num_classes = num_classes

  def forward(self, *input):
    raise NotImplementedError


class SNPDFC3(SNPDpedec):

  def __init__(self, encoder, num_features=512, num_classes=0):
    super(SNPDFC3, self).__init__(
      encoder=encoder, num_features=num_features,
      num_classes=num_classes)

    self.linear1 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    self.linear2 = utils.spectral_norm(nn.Linear(num_features, num_features))
    self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    self.l7 = utils.spectral_norm(nn.Linear(num_features, 1))
    if num_classes > 0:
      self.l_y = utils.spectral_norm(
        nn.Embedding(num_classes, num_features))

    self._initialize()

  def _initialize(self):
    init.xavier_uniform_(self.l7.weight.data)
    optional_l_y = getattr(self, 'l_y', None)
    if optional_l_y is not None:
      init.xavier_uniform_(optional_l_y.weight.data)

  def forward(self, x, y=None, decoder_only=False):
    if decoder_only:
      h = x
    else:
      h = self.encoder(x)
    h = self.linear1(h)
    h = self.relu1(h)
    h = self.linear2(h)
    h = self.relu2(h)

    output = self.l7(h)
    if y is not None:
      output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
    return output

class Logistic_Regression(nn.Module):
    def __init__(self, input_size=5000, n_labels=2):
        super(Logistic_Regression, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
