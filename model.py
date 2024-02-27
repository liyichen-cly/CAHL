from torch import nn
import torch.nn.functional as F
import torch
from dgl.nn import HeteroGraphConv, SAGEConv
import dgl.function as fn

class JKE(nn.Module):
    def __init__(self, emb_dim, k_emb, l=2):
        super().__init__()
        self.k_emb = k_emb
        self.layer_num = l
        self.pr_layers = nn.ModuleList()
        self.pr_layers.append(SAGEConv(emb_dim, 8*emb_dim, 'mean'))
        for _ in range(1, self.layer_num-1):
            self.pr_layers.append(SAGEConv(8*emb_dim, 8*emb_dim, 'mean'))
        self.pr_layers.append(SAGEConv(8*emb_dim, emb_dim, 'mean'))
        self.cc_layers = nn.ModuleList()
        self.cc_layers.append(SAGEConv(emb_dim, 8*emb_dim, 'mean'))
        for _ in range(1, self.layer_num-1):
            self.cc_layers.append(SAGEConv(8*emb_dim, 8*emb_dim, 'mean'))
        self.cc_layers.append(SAGEConv(8*emb_dim, emb_dim, 'mean'))

        self.ap_layers = nn.ModuleList()
        self.ap_layers.append(SAGEConv(emb_dim, emb_dim, 'mean'))
        for _ in range(1, self.layer_num-1):
            self.ap_layers.append(SAGEConv(emb_dim, emb_dim, 'mean'))
        self.ap_layers.append(SAGEConv(emb_dim, emb_dim, 'mean'))
        self.ap_att = APATT(emb_dim, emb_dim)


    def forward(self, g, pr_g, cc_g, sps_g, pr_ew, cc_ew):
        pr_i = self.k_emb.weight
        for layer in self.pr_layers[:-1]:
            pr_i = F.relu(layer(pr_g, pr_i, edge_weight=pr_ew))
        pr_i = self.pr_layers[-1](pr_g, pr_i, edge_weight=pr_ew)

        cc_i = self.k_emb.weight
        for layer in self.cc_layers[:-1]:
            cc_i = F.relu(layer(cc_g, cc_i, edge_weight=cc_ew))
        cc_i = self.cc_layers[-1](cc_g, cc_i, edge_weight=cc_ew)

        ap_i = self.k_emb.weight
        for layer in self.ap_layers[:-1]:
            ap_i = F.relu(layer(sps_g, ap_i))
        ap_i = self.ap_layers[-1](sps_g, ap_i)

        i = torch.stack((pr_i, cc_i), dim=1)
        i = self.ap_att(i, ap_i)
        return i

class APATT(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, z, q):
        w = torch.sum(q.unsqueeze(1)*self.project(z), dim=-1, keepdim=True).mean(0)
        beta = torch.softmax(w, dim=0)
        return (beta * z).sum(1)

class EE(nn.Module):
    def __init__(self, emb_dim, k_emb, l):
        super().__init__()
        self.k_emb = k_emb
        self.emb_dim = emb_dim
        self.layer_num = l
        self.ek_layers = nn.ModuleList()

        for _ in range(self.layer_num):
            self.ek_layers.append(HeteroGraphConv({'belong':
                                SAGEConv(emb_dim, emb_dim, 'mean')}))

        self.ee_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.ee_layers.append(HeteroGraphConv({'collaborate':
                                SAGEConv(emb_dim, emb_dim, 'mean')}))

        self.combine_fc = nn.Linear(2*self.emb_dim, self.emb_dim)
        self.act_func = nn.LeakyReLU(0.2)
        self.w1 = nn.Linear(2*self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)

        self.w4 = nn.Linear(2*self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w4.weight)
        nn.init.xavier_uniform_(self.combine_fc.weight)

    def forward(self, g, ii, cf_ew, e_emb):
        ii = self.w4(torch.cat([ii, self.k_emb.weight], dim=1))
        src = {'knowledge': ii}
        dst = {'employee': e_emb}
        for layer in self.ek_layers[:-1]:
            h_iI = F.relu(layer(g, (src, dst))['employee'])
            dst = {'employee': h_iI}
        h_iI = self.ek_layers[-1](g, (src, dst))['employee']
        h_iI = self.w2(h_iI)

        src_com = self.w1(torch.cat([h_iI, e_emb], dim=1))
        src = {'employee': src_com}
        dst = {'employee': e_emb}
        for layer in self.ee_layers[:-1]:
            h_iS = F.relu(
            layer(g, (src, dst), mod_kwargs=
                                        {'collaborate':
                                            {'edge_weight':cf_ew}})['employee'])
            dst = {'employee': h_iS}
        h_iS = self.ee_layers[-1](g, (src, dst), mod_kwargs=
                                        {'collaborate':
                                            {'edge_weight':cf_ew}})['employee']
        h_iS = self.w3(h_iS)
        h = self.act_func(self.combine_fc(torch.cat([h_iI, h_iS], dim = 1)))
        return h

class CAHL(nn.Module):
    def __init__(self, num_e, num_k, emb_dim, p_embed, k_l, e_l):
        super().__init__()
        self.num_e = num_e
        self.num_k = num_k
        self.emb_dim = emb_dim
        self.e_emb = p_embed
        self.k_emb = nn.Embedding(self.num_k, self.emb_dim)
        nn.init.xavier_uniform_(self.k_emb.weight.data)

        self.jke = JKE(self.emb_dim, self.k_emb, k_l)
        self.ee = EE(self.emb_dim, self.k_emb, e_l)

        self.gru = nn.GRUCell(self.emb_dim, self.num_k)

        self.w_m = nn.Linear(self.emb_dim, self.emb_dim)
        self.w_s = nn.Linear(self.emb_dim, self.emb_dim)
        self.w_o = nn.Linear(self.num_k, self.num_k)
        self.w_u = nn.Linear(189, self.emb_dim)

        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.xavier_uniform_(self.w_u.weight)
        self.eps = 1e-10

    def forward(self, g, pr_g, cc_g, sps_g, pr_ew, cc_ew, cf_ew):
        e_emb = self.w_u(self.e_emb)
        z = self.jke(g, pr_g, cc_g, sps_g, pr_ew, cc_ew, e_emb)
        h = self.ee(g, z, cf_ew, e_emb)

        with g.local_scope():
            funcs = {}
            h_ms = self.w_m(h)
            h_ms1 = self.w_o(torch.mm(h_ms, z.t()))
            g.nodes['employee'].data['so'] = h_ms1 * g.nodes['employee'].data['sk']
            g.nodes['employee'].data['neg_so'] = h_ms1 * (1-g.nodes['employee'].data['sk'])
            funcs['collaborate'] = (fn.u_mul_e('so', 'e', 'm'), fn.sum('m', 'x'))
            g.multi_update_all(funcs, 'sum')
            out_score = F.normalize(g.nodes['employee'].data['x'])
            funcs['collaborate'] = (fn.u_mul_e('neg_so', 'e', 'm'), fn.sum('m', 'y'))
            g.multi_update_all(funcs, 'sum')
            neg_out_score = F.normalize(g.nodes['employee'].data['y'])
            l_c = torch.mean(F.softplus(F.relu(neg_out_score)-F.relu(out_score)))
            out_score = F.normalize(self.gru(h, out_score))


        self_score = self.w_s(e_emb)
        self_score = F.normalize(torch.mm(self_score, z.t()))
        score = torch.sigmoid(out_score + self_score)
        return score, l_c