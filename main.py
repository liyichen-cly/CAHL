import os
import time
import argparse
import pickle
import numpy as np
import utils
from tqdm import tqdm

import multiprocessing
import torch
import dgl
from dgl.nn import EdgeWeightNorm
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from read_data import prepare_dataloader
from model import CAHL
from dataloader import EData
from time import strftime,localtime

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--kl', type=int, default=2, help='knowledge layer num')
parser.add_argument('--el', type=int, default=2, help='employee layer num')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=150, help='the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1234, help='the number of random seed to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
here = os.path.dirname(os.path.abspath(__file__))
CORES = multiprocessing.cpu_count() // 2

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def minmaxnorm(X):
    norm_x = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return norm_x

def read_employee_feature(n_o_eid):
    print("read employee feat...")
    with open('employee_feat.pickle', "rb") as f:
        hashid_emb = pickle.load(f)
    p_embed = []
    for i in range(len(n_o_eid)):
        p_embed.append(np.nan_to_num(np.array(hashid_emb[n_o_eid[i]])))
    p_embed = minmaxnorm(np.array(p_embed))
    return p_embed

def norm_edge_weight(train_hg, test_hg):
    norm = EdgeWeightNorm(norm='right')
    train_pr = dgl.edge_type_subgraph(train_hg, [('knowledge', 'before', 'knowledge')])
    train_pr_ew = norm(train_pr, train_pr.edata['a'].squeeze(1))

    train_cc = dgl.edge_type_subgraph(train_hg, [('knowledge', 'cooccur', 'knowledge')])
    train_cc_ew = norm(train_cc, train_cc.edata['a'].squeeze(1))

    train_hi = dgl.edge_type_subgraph(train_hg, [('employee', 'collaborate', 'employee')])
    train_cf_ew = norm(train_hi, train_hi.edata['freq'].squeeze(1))

    test_pr = dgl.edge_type_subgraph(test_hg, [('knowledge', 'before', 'knowledge')])
    test_pr_ew = norm(test_pr, test_pr.edata['a'].squeeze(1))

    test_cc = dgl.edge_type_subgraph(test_hg, [('knowledge', 'cooccur', 'knowledge')])
    test_cc_ew = norm(test_cc, test_cc.edata['a'].squeeze(1))

    test_hi = dgl.edge_type_subgraph(test_hg, [('employee', 'collaborate', 'employee')])
    test_cf_ew = norm(test_hi, test_hi.edata['freq'].squeeze(1))
    return train_pr_ew, train_cc_ew, train_cf_ew, test_pr_ew, test_cc_ew, test_cf_ew

def main():
    currentTime = strftime("%Y-%m-%d-%H-%M-%S", localtime(time.time()))
    print('Loading data...')
    now_knowledge, before1_knowledge, before2_knowledge, train_label, test_label, f_train_gt, f_test_gt, o_n_eid, n_o_eid, knowledge_id_dict, id_knowledge_dict, \
        clean1_hi, clean2_hi, train_set, test_set, train_label_dict, train_history, valid_test_history, \
            train_hg, test_hg \
    = prepare_dataloader()

    p_embed = read_employee_feature(n_o_eid)
    p_embed = torch.FloatTensor(torch.from_numpy(p_embed)).to(device)


    train_data = EData(train_set)
    test_data = EData(test_set)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    train_sk = torch.zeros((len(o_n_eid), len(knowledge_id_dict)))
    for i, s_list in enumerate(train_history):
        for x in s_list:
            train_sk[i][x] = 1

    test_sk = torch.zeros((len(o_n_eid), len(knowledge_id_dict)))
    for i, s_list in enumerate(valid_test_history):
        for x in s_list:
            test_sk[i][x] = 1

    train_hg.nodes['employee'].data['sk'] = train_sk
    test_hg.nodes['employee'].data['sk'] = test_sk

    model = CAHL(len(o_n_eid), len(knowledge_id_dict), args.embed_dim, p_embed, args.kl, args.el).to(device)

    print(train_hg)
    print(test_hg)
    train_hg = train_hg.to(device)
    test_hg = test_hg.to(device)


    train_pr_ew, train_cc_ew, train_cf_ew, test_pr_ew, test_cc_ew, test_cf_ew = norm_edge_weight(train_hg, test_hg)

    train_pr_g = train_hg.edge_type_subgraph(['before']).to(device)
    train_cc_g = train_hg.edge_type_subgraph(['cooccur']).to(device)

    test_pr_g = test_hg.edge_type_subgraph(['before']).to(device)
    test_cc_g = test_hg.edge_type_subgraph(['cooccur']).to(device)

    train_sps = dgl.metapath_reachable_graph(train_hg, ['belong', 'collaborate', 'master']).to(device)
    test_sps = dgl.metapath_reachable_graph(test_hg, ['belong', 'collaborate', 'master']).to(device)

    top_k=[1,3,5,10]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in tqdm(range(args.epoch)):
        trainForEpoch(train_hg, train_pr_g, train_cc_g, train_sps, train_pr_ew, train_cc_ew, train_cf_ew, \
                        train_loader, model, optimizer, epoch, args.epoch, criterion)
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt_dict, f'{currentTime}_latest_checkpoint.pth.tar')

    print('Load checkpoint and testing...')
    ckpt = torch.load(f'{currentTime}_latest_checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    validate(test_hg, test_pr_g, test_cc_g, test_sps, test_pr_ew, \
        test_cc_ew, test_cf_ew, test_loader, model, top_k, valid_test_history, id_knowledge_dict, True)

def trainForEpoch(train_hg, train_pr_g, train_cc_g, train_sps, train_pr_ew, train_cc_ew, train_cf_ew, train_loader, model, optimizer, epoch, num_epochs, criterion):
    model.train()
    sum_epoch_loss = 0
    start = time.time()
    for i, (uids, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        uids = uids.to(device)
        labels = labels.to(device)
        outputs, l_c = model(train_hg, train_pr_g, train_cc_g, train_sps, train_pr_ew, train_cc_ew, train_cf_ew)
        outputs = outputs[uids]
        loss = criterion(outputs, labels.float()) + l_c
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        sum_epoch_loss += loss_val
        if i % 10 == 0:
            print('[TRAIN] epoch %d/%d iter %d batch loss: %.8f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, i, loss_val, sum_epoch_loss / (i + 1),
                    len(uids) / (time.time() - start)))
        start = time.time()


def validate(valid_hg, test_pr_g, test_cc_g, test_sps, valid_pr_ew, valid_cc_ew, valid_cf_ew, \
             valid_loader, model, top_k, history_u_item, id_knowledge_dict, t=False, multicore=1):
    model.eval()
    results = {'precision': np.zeros(len(top_k)),
            'recall': np.zeros(len(top_k)),
            'ndcg': np.zeros(len(top_k))}
    if multicore == 1:
        pool = multiprocessing.Pool(4)

    users_list = []
    rating_list = []
    groundTrue_list = []
    users = 0

    all_label = np.empty((0,len(id_knowledge_dict)))
    all_prediction = np.empty((0,len(id_knowledge_dict)))
    max_K = max(top_k)
    with torch.no_grad():
        for uids, labels in valid_loader:
            exclude_index = []
            exclude_items = []
            for u_ind, u in enumerate(uids.numpy()):

                items_list = history_u_item[u]
                exclude_index.extend([u_ind] * len(items_list))
                exclude_items.extend(items_list)

            users += torch.LongTensor(uids).shape[0]
            uids = torch.LongTensor(uids).to(device)
            groundTrue = [torch.nonzero(la).squeeze(-1).tolist() for la in labels]

            preds, _ = model(valid_hg, test_pr_g, test_cc_g, test_sps, valid_pr_ew, valid_cc_ew, valid_cf_ew)
            preds = preds[uids]
            preds[exclude_index, exclude_items] = 0
            all_prediction = np.append(all_prediction, preds.cpu().numpy(), 0)
            all_label = np.append(all_label, labels.cpu().numpy(), 0)
            _, sort_index = torch.topk(preds, k=max_K)
            preds = preds.cpu().numpy()
            del preds
            users_list.append(uids.cpu())
            rating_list.append(sort_index.cpu())
            groundTrue_list.append(groundTrue)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, top_k))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(users)
        results['precision'] /= float(users)
        results['ndcg'] /= float(users)
        results['recall'] = np.round(results['recall'],6)
        results['precision'] = np.round(results['precision'],6)
        results['ndcg'] = np.round(results['ndcg'],6)
        print(f"k:{top_k}")
        if t:
            print(f"{set_color('TEST', 'yellow')} p:{results['precision']}, n:{results['ndcg']}, r:{results['recall'][-3:]}")
        else:
            print(f"p:{results['precision']}, n:{results['ndcg']}, r:{results['recall'][-3:]}")
    return results['recall'], results['precision'], results['ndcg']


def test_one_batch(X, top_k=[1,3,5,10]):
    sorted_items = X[0]
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in top_k:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k, sorted_items)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

if __name__ == '__main__':
    main()
