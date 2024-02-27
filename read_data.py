import pickle
from dgl.data.utils import load_graphs
def prepare_dataloader():
    with open('data.pkl', "rb") as f:
        (now_knowledge, before1_knowledge, before2_knowledge, train_label, test_label, f_train_gt, f_test_gt) = pickle.load(f)
        (o_n_eid, n_o_eid, knowledge_id_dict, id_knowledge_dict) = pickle.load(f)
        (clean1_hi, clean2_hi) = pickle.load(f)
        (train_set, test_set, train_label_dict, train_history, valid_test_history) = pickle.load(f)
    print("read pickle over..")
    hg = load_graphs('e_k_graph.bin')
    train_hg = hg[0][0]
    test_hg = hg[0][1]
    return now_knowledge, before1_knowledge, before2_knowledge, train_label, test_label, f_train_gt, f_test_gt, o_n_eid, n_o_eid, knowledge_id_dict, id_knowledge_dict, \
        clean1_hi, clean2_hi, train_set, test_set, train_label_dict, train_history, valid_test_history, \
            train_hg, test_hg
