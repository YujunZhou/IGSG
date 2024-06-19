from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import utils
from torch.nn import init, TransformerEncoder, TransformerEncoderLayer
import math


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.num_catfeatures = 32
        self.num_confeatures = 9
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_catfeatures * self.num_cate,
            embedding_dim=64,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.hidden_size = 64
        self.num_classes = 2
        self.mlp_cat = nn.Sequential(
            nn.Linear(self.num_catfeatures * self.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(self.num_confeatures, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.classify = nn.Linear(2*self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.num_catfeatures * self.num_cate))

    def forward(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_cat = x_cat.reshape(x_cat.size(0), -1)
        x_cat = self.mlp_cat(x_cat)
        x_con = self.mlp_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
        x = self.classify(x)
        x = self.softmax(x)
        return x

    def empty_emb(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list

    def valid_matrix(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        valid_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    valid_list.append(0)
                else:
                    valid_list.append(1)
        valid_mat = torch.LongTensor(valid_list).reshape(self.num_catfeatures, self.num_cate)
        return valid_mat

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def embed(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_cat = x_cat.reshape(x_cat.size(0), -1)
        x_cat = self.mlp_cat(x_cat)
        x_con = self.mlp_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
        return x



class Transformer(nn.Module):
    def __init__(self, num_layers=2, embedding_dim=16, hidden_dim=16, num_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_catfeatures = 32
        self.num_confeatures = 9
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_catfeatures * self.num_cate,
            embedding_dim=embedding_dim,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        num_classes = 2
        self.emb_con = nn.Linear(1, embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.model_input = torch.LongTensor(range(self.num_catfeatures * self.num_cate))
        # self.projection = nn.Sequential(
        #     nn.Linear(embedding_dim, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )

    def forward(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_con = x_con.unsqueeze(dim=2)
        x_con = self.emb_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
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

    def empty_emb(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list

    def valid_matrix(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        valid_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    valid_list.append(0)
                else:
                    valid_list.append(1)
        valid_mat = torch.LongTensor(valid_list).reshape(self.num_catfeatures, self.num_cate)
        return valid_mat

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
    def __init__(self, num_layers=2, embedding_dim=16, hidden_dim=16, num_heads=8, dropout=0.1):
        super(Transformer_E, self).__init__()
        self.num_catfeatures = 32
        self.num_confeatures = 9
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_catfeatures * self.num_cate,
            embedding_dim=embedding_dim,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.emb_con = nn.Linear(1, embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.model_input = torch.LongTensor(range(self.num_catfeatures * self.num_cate))
        # self.projection = nn.Sequential(
        #     nn.Linear(embedding_dim, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )
        self.hidden_size = embedding_dim

    def forward(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_con = x_con.unsqueeze(dim=2)
        x_con = self.emb_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x)
        output = output.transpose(0, 1)  # output.shape = (batch_size, seq_len, embedding_dim)
        output = torch.sum(output, dim=1)
        # energy = self.projection(output)
        # weights = F.softmax(energy.squeeze(-1), dim=1)
        # # (B, L, H) * (B, L, 1) -> (B, H)
        # output = (output * weights.unsqueeze(-1)).sum(dim=1)
        return output

    def empty_emb(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list


class Transformer_Dc(nn.Module):
    def __init__(self, E, embedding_dim=16):
        super(Transformer_Dc, self).__init__()
        num_classes = 2
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.E = E

    def forward(self, x_con, x_cat):
        output = self.E(x_con, x_cat)
        output = self.fc(output)
        return output

class MLP_nobn(nn.Module):
    def __init__(self):
        super(MLP_nobn, self).__init__()
        self.num_catfeatures = 32
        self.num_confeatures = 9
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_catfeatures * self.num_cate,
            embedding_dim=64,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.hidden_size = 64
        self.num_classes = 2
        self.mlp_cat = nn.Sequential(
            nn.Linear(self.num_catfeatures * self.hidden_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(self.num_confeatures, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.classify = nn.Linear(2*self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.num_catfeatures * self.num_cate))

    def forward(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_cat = x_cat.reshape(x_cat.size(0), -1)
        x_cat = self.mlp_cat(x_cat)
        x_con = self.mlp_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
        x = self.classify(x)
        x = self.softmax(x)
        return x

    def empty_emb(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list

    def valid_matrix(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        valid_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    valid_list.append(0)
                else:
                    valid_list.append(1)
        valid_mat = torch.LongTensor(valid_list).reshape(self.num_catfeatures, self.num_cate)
        return valid_mat

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
        self.num_catfeatures = 32
        self.num_confeatures = 9
        self.num_cate = 52
        self.embedding = nn.Embedding(
            num_embeddings=self.num_catfeatures * self.num_cate,
            embedding_dim=64,
            max_norm=1.0
        )
        empty_list = self.empty_emb()
        self.embedding.weight.data[empty_list] = 0
        self.hidden_size = 64
        self.mlp_cat = nn.Sequential(
            nn.Linear(self.num_catfeatures * self.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(self.num_confeatures, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.model_input = torch.LongTensor(range(self.num_catfeatures * self.num_cate))

    def forward(self, x_con, x_cat):
        model_input = self.model_input.reshape(1, -1).cuda()
        weight = self.embedding(model_input)
        x_cat = x_cat.flatten(start_dim=1, end_dim=2)
        x_cat = torch.unsqueeze(x_cat, dim=2)
        x_cat = x_cat * weight
        x_cat = x_cat.reshape(x_cat.size(0), self.num_catfeatures, self.num_cate, -1)
        x_cat = torch.sum(x_cat, dim=2)
        x_cat = x_cat.reshape(x_cat.size(0), -1)
        x_cat = self.mlp_cat(x_cat)
        x_con = self.mlp_con(x_con)
        x = torch.cat((x_con, x_cat), dim=1)
        return x

    def empty_emb(self):
        census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                           8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

        empty_list = []
        for i in range(len(census_category)):
            for j in range(52):
                if j >= census_category[i]:
                    empty_list.append(i * 52 + j)
        return empty_list


class MLP_Dc(nn.Module):
    def __init__(self, E):
        super(MLP_Dc, self).__init__()
        self.hidden_size = 64
        self.num_classes = 2
        self.classify = nn.Linear(2 * self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.E = E

    def forward(self, x_con, x_cat):
        x = self.E(x_con, x_cat)
        x = self.classify(x)
        x = self.softmax(x)
        return x


class SNPDCensus(nn.Module):

    def __init__(self, encoder, num_features, num_classes=0):
        super(SNPDCensus, self).__init__()
        self.encoder = encoder
        self.num_features = num_features
        self.num_classes = num_classes

    def forward(self, *input):
        raise NotImplementedError


class SNPDFC3(SNPDCensus):

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
            h = self.encoder(x[0], x[1])
        h = self.linear1(h)
        h = self.relu1(h)
        h = self.linear2(h)
        h = self.relu2(h)

        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output