
import torch
import numpy as np
from utils import all_metrics, print_metrics, write_excel
from sklearn import metrics


def train(args, model, optimizer, epoch, gpu, data_loader, adj):
    losses = []

    model.train()
    # entity_ids = torch.LongTensor(entity_ids)     # 将实体id转化为tensor
    entity_ids = torch.arange(args.entity_num)

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):

        inputs_id, labels, text_inputs = next(data_iter)

        inputs_id, labels = torch.LongTensor(inputs_id), torch.FloatTensor(labels)
        if gpu >= 0:
            inputs_id, labels, entity_ids = inputs_id.cuda(gpu), labels.cuda(gpu), entity_ids.cuda(gpu)
            adj = adj.cuda(gpu)

        output, loss = model(inputs_id, labels, adj, entity_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def test(args, model, fold, dicts, data_loader, adj, icd_code, logger):

    # filename = data_path.replace('train', fold)
    logger.info('file for evaluation: %s' % fold)
    num_labels = len(dicts['ind2c'])

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []

    model.eval()
    entity_ids = torch.arange(args.entity_num)

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    for i in range(num_iter):
        with torch.no_grad():

            inputs_id, labels, text_inputs = next(data_iter)

            inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

            if args.gpu >= 0:
                inputs_id, labels = inputs_id.cuda(args.gpu), labels.cuda(args.gpu)
                adj, entity_ids = adj.cuda(args.gpu), entity_ids.cuda(args.gpu)
            output, loss = model(inputs_id, labels, adj, entity_ids)

            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

            losses.append(loss.item())
            target_data = labels.data.cpu().numpy()

            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    report = metrics.classification_report(y, yhat, target_names=icd_code, digits=4)

    # k = 5 if num_labels == 50 else [8, 15]
    k = [1, 5, 8]
    metrics_ = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    print_metrics(metrics_, logger)
    # write_excel(metrics_, args)
    if fold == 'test':
        write_excel(metrics_, args)
    # metrics_['loss_%s' % fold] = np.mean(losses)
    return metrics_, report
