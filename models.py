import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from utils import load_embeddings
from math import floor
from GCNLayer import GraphConvolution
# device = torch.device('cpu')


class WordRep(nn.Module):
    def __init__(self, args, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        if args.graph_embed_file:
            print("loading graph ent pretrained embeddings from {}".format(args.graph_embed_file))
            W = torch.Tensor(load_embeddings(args.graph_embed_file))

            self.embedding_graph = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embedding_graph.weight.data = W.clone()
        else:
            self.embedding_graph = nn.Embedding(args.entity_num, args.embed_size)

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [args.num_filter_maps, args.num_filter_maps],
                     2: [args.num_filter_maps, 100, args.num_filter_maps],
                     3: [args.num_filter_maps, 100, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }

    def forward(self, x, node_ids=[]):
        x = self.embed(x)
        x = self.embed_drop(x)

        if node_ids != None:
            node_embed = self.embedding_graph(node_ids)
            return x, node_embed
        return x


class OutputLayer(nn.Module):
    def __init__(self, args, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, args.class_nums)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, args.class_nums)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, x, target, text_inputs):

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss


class CNN(nn.Module):
    def __init__(self, args, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, dicts)

        filter_size = int(args.filter_size)

        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        attn_d = 256

        self.w = nn.Parameter(torch.FloatTensor(attn_d, args.num_filter_maps))  # da x embed_size
        xavier_uniform(self.w)
        self.u = nn.Parameter(torch.FloatTensor(args.class_nums, attn_d))  # L x da
        xavier_uniform(self.u)

    def forward(self, x):
        # x:
        Z = torch.tanh(torch.matmul(self.w, x))            # [batch_size,seq_len,attn_d]
        A = torch.softmax(torch.matmul(self.u, Z), dim=2)  # [batch_size,seq_len,labels_num]
        V = torch.matmul(x, A.transpose(1, 2))             # [batch_size,labels_num,filter_num*feature_size]
        V = V.transpose(1, 2)

        return V


class MultiCNN(nn.Module):
    def __init__(self, args, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)

            self.attn = AttentionLayer(args)

        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

            self.attn = nn.ModuleList()
            for num in range(self.filter_num):
                attention = AttentionLayer(args)
                self.attn.add_module('attn-{}'.format(num), attention)

        ffn_size = 128

        self.final1 = nn.Linear(args.num_filter_maps*self.filter_num, ffn_size)
        xavier_uniform(self.final1.weight)
        self.final2 = nn.Linear(ffn_size, args.class_nums)
        xavier_uniform(self.final2.weight)
        self.loss_function = nn.BCEWithLogitsLoss()
        # self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

    def forward(self, x, target, text_inputs, adj, entity_ids):

        x = self.word_rep(x)
        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
            V = self.attn(x)
        else:
            conv_result, attn_result = [], []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))

            for i, attn in enumerate(self.attn):
                x = conv_result[i].transpose(1, 2)
                attn_result.append(attn(x))

            V = torch.cat(attn_result, dim=2)

        # output layer
        V = torch.relu(self.final1(V))
        y_hat = self.final2.weight.mul(V).sum(dim=2).add(self.final2.bias)

        loss = self.loss_function(y_hat, target)

        return y_hat, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
        )

        self.use_res = use_res
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += x
        out = torch.tanh(out)
        out = self.dropout(out)
        return out


class MSLAN_ACG(nn.Module):
    def __init__(self, args, dicts):
        super(MSLAN_ACG, self).__init__()
        self.word_rep = WordRep(args, dicts)
        self.class_nums = args.class_nums
        self.encoders = nn.ModuleList()
        self.num_entity = args.entity_num

        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            encoder = nn.ModuleList()

            # Convolution Layer
            tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            encoder.add_module('baseconv', tmp)

            # Batch Normalization
            norm = nn.BatchNorm1d(args.num_filter_maps)
            encoder.add_module('norm', norm)

            # Residual Block
            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                encoder.add_module('resconv-{}'.format(idx), tmp)

            # Label Attention
            attention = AttentionLayer(args)
            encoder.add_module('labelattn', attention)

            self.encoders.add_module('encoder-{}'.format(filter_size), encoder)

        # GCN
        self.gc1 = GraphConvolution(args.embed_size, args.embed_size)
        self.gc2 = GraphConvolution(args.embed_size, args.embed_size)

        # Output Layer
        ffn_size = 128
        self.final1 = nn.Linear(args.num_filter_maps * self.filter_num + args.embed_size, ffn_size)
        xavier_uniform(self.final1.weight)
        self.final2 = nn.Linear(ffn_size, args.class_nums)
        xavier_uniform(self.final2.weight)

        # Loss function
        self.loss_function = nn.BCEWithLogitsLoss()

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x, target, adj, entity_ids):

        x, node_embedding = self.word_rep(x, entity_ids)       # embedding

        x = x.transpose(1, 2)

        encoder_result = []
        for encoder in self.encoders:
            tmp = x
            for idx, md in enumerate(encoder):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            encoder_result.append(tmp)

        V = torch.cat(encoder_result, dim=2)    # concat encoder results

        # GCN
        out_graph = F.relu(self.gc1(node_embedding, adj))
        out_graph = self.dropout(out_graph)
        out_graph = self.gc2(out_graph, adj)
        out_graph = out_graph[:self.class_nums, :].unsqueeze(0).expand(V.size(0), -1, -1)

        out = torch.cat((out_graph, V), dim=2)

        # output layer
        V = torch.relu(self.final1(out))
        y_hat = self.final2.weight.mul(V).sum(dim=2).add(self.final2.bias)
        y_hat = self.dropout(y_hat)

        loss = self.loss_function(y_hat, target)

        return y_hat, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


def pick_model(args, dicts):
    if args.model_name == 'CNN':
        model = CNN(args, dicts)
    elif args.model_name == 'MultiCNN':
        model = MultiCNN(args, dicts)
    elif args.model_name == 'MVCLA_ACG':
        model = MVCLA_ACG(args, dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model = model.cuda(args.gpu)

    return model
