import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./results/models')
parser.add_argument('-LOG_DIR', type=str, default='./results/log/')
parser.add_argument('-DATA_DIR', type=str, default='./data')
parser.add_argument('-MIMIC_3_DIR', type=str, default='./data/mimic3')
parser.add_argument('-CHINESE_DIR', type=str, default='./data/chineseEMR')


parser.add_argument("-train_path", type=str, default='./data/mimic3/train_50.csv')
parser.add_argument("-dev_path", type=str, default='./data/mimic3/dev_50.csv')
parser.add_argument("-test_path", type=str, default='./data/mimic3/test_50.csv')
parser.add_argument("-graph_path", type=str, default='./data/mimic3/graph_mimic3-50_co-occur-100_.npy')
parser.add_argument("-vocab_path", type=str, default='./data/mimic3/vocab.csv')
parser.add_argument("-graph_vocab_path", type=str, default='./data/mimic3/graph_vocab.csv')
parser.add_argument("-label_path", type=str, default='./data/mimic3/TOP_50_CODES.csv')

parser.add_argument('-entity_num', default=0, type=int, help='the entity number of Knowledge Graph')
parser.add_argument("-version", type=str, choices=['mimic3', 'chinese'], default='mimic3')
parser.add_argument("-MAX_LENGTH", type=int, default=4000)
parser.add_argument("-class_nums", type=int, default=50)

# model
parser.add_argument("-model_name", type=str, choices=['CNN', 'MultiCNN', 'MVCLA_ACG', 'bert_seq_cls'], default='MVCLA_ACG')
parser.add_argument("-filter_size", type=str, default="3,5,9")
parser.add_argument("-num_filter_maps", type=int, default=100)
parser.add_argument("-conv_layer", type=int, default=2)
parser.add_argument("-embed_file", type=str, default='./data/mimic3/processed_full.embed')
parser.add_argument("-graph_embed_file", type=str, default='./data/mimic3/processed_full_ent_new.embed')
parser.add_argument('-excel_path', type=str, default='./results/mimic3_result.xls')
parser.add_argument("-embed_size", type=int, default=100)
parser.add_argument("-test_model", type=str, default=None)

# training
parser.add_argument("-n_epochs", type=int, default=500)
parser.add_argument("-dropout", type=float, default=0.3)
parser.add_argument("-patience", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-criterion", type=str, default='f1_micro', choices=['prec_at_8', 'f1_micro', 'prec_at_5'])
parser.add_argument("-gpu", type=int, default=2, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-tune_wordemb", action="store_const", const=True, default=False)
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize model, other if fix the seed')
parser.add_argument('-PLM', type=int, default=0,  help='1 is using PLM, 0 is not using')
parser.add_argument('-use_word', default=True, action='store_true')

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
