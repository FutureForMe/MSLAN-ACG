import os
import csv
import sys
import time
import torch
import random
import datetime
import pprint
import numpy as np
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader

from options import args
from models import pick_model
from train_test import train, test
from utils import load_lookups, prepare_instance, MyDataset, my_collate,\
    early_stop, save_everything, load_graph_data, create_logger

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = True

    logger = create_logger(
        args.LOG_DIR + args.model_name +
        datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S") +
        '_seed_' + str(args.random_seed) + '.log')

    '''Logging hyper parameters'''
    logger.info("Training with \n{}\n".format(pprint.pformat(args.__dict__, indent=4)))

    logger.info('Load label and vocab...')
    dicts, icd_code = load_lookups(args)

    args.class_nums = len(dicts["c2ind"])

    '''Load graph'''
    logger.info('Load graph...')
    adj = load_graph_data(args, dicts)

    '''Train and test model'''
    if args.test_model is not None:
        model = pick_model(args, dicts)

        test_instances = prepare_instance(dicts, args.test_path, args, args.MAX_LENGTH)
        test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate)
        print("test_instances {}".format(len(test_instances)))

        test_model_path = args.test_model
        sd = torch.load(test_model_path)
        model.load_state_dict(sd)

        if args.gpu >= 0:
            model = model.cuda()

        metrics_te, report = test(args, model, "test", dicts, test_loader, adj, icd_code, logger)
        print(report)

    else:
        model = pick_model(args, dicts)

        if not args.test_model:
            optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        else:
            optimizer = None

        if args.tune_wordemb == False:
            model.freeze_net()

        metrics_hist = defaultdict(lambda: [])
        metrics_hist_te = defaultdict(lambda: [])
        metrics_hist_tr = defaultdict(lambda: [])

        # Load train dataset
        logger.info('Load dataset {} ...'.format(args.train_path))
        train_instances = prepare_instance(dicts, args.train_path, args, args.MAX_LENGTH)
        train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=my_collate)

        # Load dev dataset
        logger.info('Load dataset {} ...'.format(args.dev_path))
        dev_instances = prepare_instance(dicts, args.dev_path, args, args.MAX_LENGTH)
        dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=my_collate)

        # Load test dataset
        logger.info('Load dataset {} ...'.format(args.test_path))
        test_instances = prepare_instance(dicts, args.test_path, args, args.MAX_LENGTH)
        test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=my_collate)

        test_flag = False

        # Begin training
        for epoch in range(args.n_epochs):
            logger.info('Epoch %d' % epoch)
            if epoch == 0:
                model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model_name,
                                                                   time.strftime('%b_%d_%H_%M_%S', time.localtime()),
                                                                   'seed', str(args.random_seed)]))
                os.makedirs(model_dir)

            # train
            epoch_start = time.time()
            losses = train(args, model, optimizer, epoch, args.gpu, train_loader, adj)
            loss = np.mean(losses)
            epoch_finish = time.time()
            logger.info("epoch finish in train dataset, time: %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))

            if epoch == args.n_epochs - 1:
                logger.info("last epoch: testing on dev and test sets")
                test_flag = True

            # dev
            evaluation_start = time.time()
            metrics, report = test(args, model, "dev", dicts, dev_loader, adj, icd_code, logger)     # 在验证集上进行测试
            evaluation_finish = time.time()
            logger.info("evaluation finish in dev dataset, time: %.2fs" % (evaluation_finish - evaluation_start))

            # if early stop, save model
            if args.criterion in metrics_hist.keys():
                if early_stop(metrics_hist, args.criterion, args.patience):
                    # stop training, do tests on test and train sets, and then stop the script
                    logger.info("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                    test_flag = True

            # test
            if test_flag:
                logger.info('Loading the best model in dev dataset ...')
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = pick_model(args, dicts)
                logger.info('The best result in dev dataset:')
                _, report_dev = test(args, model, "dev", dicts, dev_loader, adj, icd_code, logger)
                logger.info('The best result in test dataset:')
                metrics_te, report = test(args, model, "test", dicts, test_loader, adj, icd_code, logger)
            else:
                metrics_te = defaultdict(float)
            metrics_tr = {'loss': loss}
            metrics_all = (metrics, metrics_te, metrics_tr)

            # save result
            for name in metrics_all[0].keys():
                metrics_hist[name].append(metrics_all[0][name])
            for name in metrics_all[1].keys():
                metrics_hist_te[name].append(metrics_all[1][name])
            for name in metrics_all[2].keys():
                metrics_hist_tr[name].append(metrics_all[2][name])
            metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

            save_everything(args, metrics_hist_all, model, model_dir, logger, args.criterion, report, test_flag)

            sys.stdout.flush()

            if test_flag:
                break
