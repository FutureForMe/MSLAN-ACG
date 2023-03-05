import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict
import scipy.sparse as sp
from torch.utils.data import Dataset
# from elmo.elmo import batch_to_ids
import xlrd
import xlwt
from xlutils.copy import copy
import json
import torch
import logging
from sklearn.metrics import roc_curve, auc
from transformers import AutoModel, AutoTokenizer
UNK, PAD, CLS = '**UNK**', '**PAD**', '[CLS]'


def load_vocab_dict(args, vocab_file):
    vocab = set()

    with open(vocab_file, 'r', encoding='utf-8') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())

    ind2w = {i: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}

    return ind2w, w2ind


def load_lookups(args):
    ind2w, w2ind = load_vocab_dict(args, args.vocab_path)   # load vocab
    codes = []
    with open(args.label_path, 'r') as labelfile:
        lr = csv.reader(labelfile)
        for i, row in enumerate(lr):
            codes.append(row[0])
    codes.sort()
    ind2c = {i: c for i, c in enumerate(codes)}  # sort code

    icd_code = [0] * len(ind2c)
    for i, c in ind2c.items():
        icd_code[i] = c

    c2ind = {c: i for i, c in ind2c.items()}

    ents = []
    with open(args.graph_vocab_path, "r") as graph_ent:
        ent = csv.reader(graph_ent)
        for i, row in enumerate(ent):
                ents.append(row[0])
    e2ind = {c: i for i,c in enumerate(ents)}

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'e2ind': e2ind}

    return dicts, icd_code


def prepare_instance(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    if args.PLM:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
    else:
        if args.use_word:
            tokenizer = lambda x: x.split(' ')
        else:
            tokenizer = lambda x: [y for y in x]

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)    # header

        for row in r:
            if args.version == 'mimic3':
                text = row[2]
                labels = row[3].split(';')
            elif args.version == 'chinese':
                text = row[4] + row[5] + row[6] + row[7] + row[8] + row[9]
                labels = row[-2].split(';')

            labels_idx = np.zeros(num_labels)
            labelled = False

            for l in labels:
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue

            tokens_ = tokenizer(text)
            # tokens_ = text.split()
            tokens = []
            tokens_id = []

            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                tokens.append(token)
                # token_id = w2ind.get(token, w2ind.get(UNK))
                token_id = w2ind[token] if token in w2ind else len(w2ind) # + 1
                tokens_id.append(token_id)

            # tokens, tokens_id, mask, seq_len = get_tokens_and_ids(max_length, args.PLM, tokenizer, text, vocab)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]

            dict_instance = {'label': labels_idx,
                             'tokens': tokens,
                             "tokens_id": tokens_id}

            instances.append(dict_instance)

    return instances


def load_graph_data(args, dicts):
    all_graph = np.load(args.graph_path, allow_pickle=True)
    w2ind, c2ind, ind2c, e2ind = dicts['w2ind'], dicts['c2ind'], dicts['ind2c'], dicts['e2ind']

    args.entity_num = len(e2ind)

    edges = []
    for i in range(len(all_graph)):
        if args.version == 'mimic3':
            head_id = e2ind[all_graph[i][0][0].split("..")[0]]
            tail_id = e2ind[all_graph[i][0][1].split("..")[0]]
        else:
            head_id = e2ind[all_graph[i][0][0]]
            tail_id = e2ind[all_graph[i][0][1]]
        edges.append([head_id, tail_id])
        # edges.append([tail_id, head_id])
    edges = np.array(edges)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(args.entity_num, args.entity_num),
                        dtype=np.float32)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj.todense())

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=np.int):

    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x


def load_embeddings(embed_file):
    # also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)

        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W


def my_collate(x):

    words = [x_['tokens_id'] for x_ in x]

    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)

    labels = [x_['label'] for x_ in x]

    text_inputs = [x_['tokens'] for x_ in x]

    text_inputs_id = [x_['tokens_id'] for x_ in x]

    return inputs_id, labels, text_inputs_id


def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev':
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False


def save_metrics(args, metrics_hist_all, model_dir, report):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        # concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update({"%s_te" % (name):val for (name,val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_all[2].items()})
        data.update(args.__dict__)
        json.dump(data, metrics_file, indent=1)
    
    with open(model_dir + "/report.txt", "w") as report_file:
        report_file.write(report)


def save_everything(args, metrics_hist_all, model, model_dir, logger, criterion, report, evaluate=False):

    save_metrics(args, metrics_hist_all, model_dir, report)

    if not evaluate:
        # save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[0][criterion])):
            if criterion == 'loss_dev':
                eval_val = np.nanargmin(metrics_hist_all[0][criterion])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][criterion])

            if eval_val == len(metrics_hist_all[0][criterion]) - 1:
                sd = model.cpu().state_dict()
                torch.save(sd, model_dir + "/model_best_%s.pth" % criterion)
                if args.gpu >= 0:
                    model.cuda(args.gpu)
    logger.info("saved metrics, params, model to directory %s\n" % (model_dir))


def print_metrics(metrics, logger):
    if "auc_macro" in metrics.keys():
        logger.info("[MACRO] accuracy, precision, recall, f-measure, AUC")
        logger.info("        %.4f,   %.4f,    %.4f, %.4f,    %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        logger.info("[MACRO] accuracy, precision, recall, f-measure")
        logger.info("        %.4f,   %.4f,    %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        logger.info("[MICRO] accuracy, precision, recall, f-measure, AUC")
        logger.info("        %.4f,   %.4f,    %.4f, %.4f,    %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        logger.info("[MICRO] accuracy, precision, recall, f-measure")
        logger.info("        %.4f,   %.4f,    %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            logger.info("%s: %.4f" % (metric, val))


def write_excel(metrics, args):
    workbook = xlrd.open_workbook(args.excel_path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格

    new_worksheet.write(rows_old, 0, args.model_name)
    new_worksheet.write(rows_old, 1, args.lr)
    new_worksheet.write(rows_old, 2, args.batch_size)
    new_worksheet.write(rows_old, 3, args.filter_size)
    new_worksheet.write(rows_old, 4, args.random_seed)

    if "auc_macro" in metrics.keys():
        new_worksheet.write(rows_old, 5, metrics["acc_macro"])
        new_worksheet.write(rows_old, 6, metrics["prec_macro"])
        new_worksheet.write(rows_old, 7, metrics["rec_macro"])
        new_worksheet.write(rows_old, 8, metrics["f1_macro"])
        new_worksheet.write(rows_old, 9, metrics["auc_macro"])

    if "auc_micro" in metrics.keys():
        new_worksheet.write(rows_old, 10, metrics["acc_micro"])
        new_worksheet.write(rows_old, 11, metrics["prec_micro"])
        new_worksheet.write(rows_old, 12, metrics["rec_micro"])
        new_worksheet.write(rows_old, 13, metrics["f1_micro"])
        new_worksheet.write(rows_old, 14, metrics["auc_micro"])

    col = 15
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            new_worksheet.write(rows_old, col, val)
            col += 1

    new_workbook.save(args.excel_path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)


def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    # macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    # micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc


def recall_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    # macro
    macro = all_macro(yhat, y)

    # micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    # AUC and @k
    if yhat_raw is not None and calc_auc:
        # allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics


def create_logger(logger_file_name):
    logger = logging.getLogger()    # 设定日志对象
    logger.setLevel(logging.INFO)   # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger





