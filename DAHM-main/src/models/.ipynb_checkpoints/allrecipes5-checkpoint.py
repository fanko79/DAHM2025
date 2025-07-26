import os
import numpy as np
import scipy.sparse as sp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class ALLRECIPES5(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ALLRECIPES5, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        # text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # edge prune
        self.edge_indices, self.edge_values = self.get_edge_info()

        self.masked_adj = None
        self.forward_adj = None
        self.pruning_random = config["purning_random"]
        self.self_loop = config["self_loop"]
        self.dropout = config["dropout"]

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.hyper_num = config["hyper_num"]
        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = config['hyper_tau']

        self.image_preference = nn.Embedding(self.n_users, self.embedding_dim)
        self.text_preference = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.image_preference.weight)
        nn.init.xavier_uniform_(self.text_preference.weight)

        self.n_m_layers = config['n_m_layers']
        self.agg = config["agg"]
        self.m_agg = config["m_agg"]
        self.transform = nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        self.cl_tau = config["cl_tau"]
        self.cl_loss_type = config["cl_loss_type"]
        self.purifer = config["purifer"]
        self.reg_loss = config['reg_loss']

        if self.v_feat is not None:
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))
        self.keep_rate = config["keep_rate"]
        self.drop = nn.Dropout(p=1 - self.keep_rate)
        self.alpha = config["alpha"]
        self.n_hyper_layers = config["n_hyper_layers"]
        self.lsim = config['lsim']
        self.batch_size = config['train_batch_size']
        self.cl_loss2 = config['cl_loss2']
        self.switch = config['switch']
        self.d = config['d']

    def pre_epoch_processing(self):
        if self.dropout == 0.0:
            self.masked_adj = self.norm_adj # self.norm_adj_matrix
            return
        keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
        if self.pruning_random:
            # pruning randomly
            keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len))
        else:
            # pruning edges by pro
            alpha = torch.abs(self.edge_values)
            if self.d:
                dirichlet_dist = Dirichlet(alpha)
                sampled_values = dirichlet_dist.sample()
                self.edge_values = sampled_values.squeeze()
            keep_idx = torch.multinomial(self.edge_values, keep_len)  # prune high-degree nodes
        if self.switch == True:
            self.pruning_random = True ^ self.pruning_random
        keep_indices = self.edge_indices[:, keep_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))  # 双向性
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=2048):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.cl_tau)  #

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                        refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                               i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.cl_tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.cl_tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))

        values_file = os.path.join(self.dataset_path, 'values.txt')
        def save_values_to_file(values_list, filename):
            with open(filename, 'w') as file:
                for value in values_list:
                    file.write(f"{value}\n")
        if not os.path.exists(values_file):
            save_values_to_file(values.tolist(), values_file)
        
        return edges, values

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if self.self_loop == 'loop':
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]

        degree_file = os.path.join(self.dataset_path, 'degree.txt')
        def save_degree_to_file(degree_list, filename):
            with open(filename, 'w') as f:
                for degree in degree_list:
                    f.write(f"{degree}\n")
        if not os.path.exists(degree_file):
            degrees = np.array(adj_mat.sum(1)).flatten()
            save_degree_to_file(degrees, degree_file)

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def nonzero_idx(self):
        r, c = self.R.nonzero()
        idx = list(zip(r, c))
        return idx

    def normalize_laplacian(self, edge_index, edge_weight):
        num_nodes = maybe_num_nodes(edge_index)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_weight

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight) # I,dm → I,64
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight) # I,dm → I,64

        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper) # I, dm → I, hyper_num
            uv_hyper = torch.mm(self.R, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.R, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        image_preference = self.image_preference.weight  # [U, 64]
        text_preference = self.text_preference.weight

        if self.purifer == 'ori':
            image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats)) # I,64
            text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))
        elif self.purifer == 'non':
            image_item_embeds = self.gate_v(image_feats)
            text_item_embeds = self.gate_t(text_feats)
        elif self.purifer == 'blank':
            image_item_embeds = image_feats
            text_item_embeds = text_feats
            
        image_item_embeds = F.normalize(image_item_embeds) # I,64
        text_item_embeds = F.normalize(text_item_embeds)

        # propagate
        ego_image_emb = torch.cat([image_preference, image_item_embeds], dim=0)  # [U+I-txt, d]
        ego_text_emb = torch.cat([text_preference, text_item_embeds], dim=0)  # [U+I-img, d]

        all_image_emb = [ego_image_emb]
        all_text_emb = [ego_text_emb]
        for layer in range(self.n_m_layers):
            side_image_emb = torch.sparse.mm(adj, ego_image_emb)
            ego_image_emb = side_image_emb
            all_image_emb += [ego_image_emb]

            side_text_emb = torch.sparse.mm(adj, ego_text_emb)
            ego_text_emb = side_text_emb
            all_text_emb += [ego_text_emb]

        if self.m_agg == "last":
            final_image_preference, final_image_emb = torch.split(
                ego_image_emb, [self.n_users, self.n_items], dim=0
            ) # [U, d] [I, d]
            final_text_preference, final_text_emb = torch.split(
                ego_text_emb, [self.n_users, self.n_items], dim=0
            )
        elif self.m_agg == "mean":
            mean_all_image_emb = torch.mean(torch.stack(all_image_emb), dim=0)
            final_image_preference, final_image_emb = torch.split(mean_all_image_emb, [self.n_users, self.n_items], dim=0)
            mean_all_text_emb = torch.mean(torch.stack(all_text_emb), dim=0)
            final_text_preference, final_text_emb = torch.split(mean_all_text_emb, [self.n_users, self.n_items], dim=0)

        # User-Item View
        item_embeds = self.item_id_embedding.weight # I, d
        user_embeds = self.user_embedding.weight # U, d
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0) # U+I, d
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings # Eid,u Eid,i
        final_id_preference, final_id_emb = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        # [U, d] [I, d]
        
        for i in range(self.n_hyper_layers):
            # hyper_image_item_embeds = torch.sparse.mm(self.image_original_adj, self.drop(iv_hyper)) # [I, I] * [I, hyper_num]
            # hyper_text_item_embeds = torch.sparse.mm(self.text_original_adj, self.drop(it_hyper))
            lat = self.drop(iv_hyper) # I, hyper_num
            hyper_image_item_embeds = torch.mm(lat.T, final_id_emb) # hyper_num, 64
            hyper_image_item_embeds = torch.mm(lat, hyper_image_item_embeds) # I, d
            lat = self.drop(it_hyper)  # I, hyper_num
            hyper_text_item_embeds = torch.mm(lat.T, final_id_emb)  # hyper_num, 64
            hyper_text_item_embeds = torch.mm(lat, hyper_text_item_embeds)
        
        hyper_image_user_embeds = torch.sparse.mm(self.R, hyper_image_item_embeds) # [U, I]*[I, d]
        hyper_image_embeds = torch.cat([hyper_image_user_embeds, hyper_image_item_embeds], dim=0) # [U+I, d]

        hyper_text_user_embeds = torch.sparse.mm(self.R, hyper_text_item_embeds)
        hyper_text_embeds = torch.cat([hyper_text_user_embeds, hyper_text_item_embeds], dim=0)

        att_common = torch.cat([self.query_common(hyper_image_embeds), self.query_common(hyper_text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * hyper_image_embeds + weight_common[:, 1].unsqueeze(dim=1) * hyper_text_embeds # [U+I, d]
        common_embeds, noise_embeds = self.denoise(common_embeds, content_embeds)
        all_h_image_embeds = hyper_image_embeds - common_embeds # + noise_embeds
        all_h_text_embeds = hyper_text_embeds - common_embeds # + noise_embeds

        image_prefer = self.gate_image_prefer(content_embeds) # U+I, d final_id_preference, final_id_emb
        text_prefer = self.gate_text_prefer(content_embeds)

        final_h_image_embeds = all_h_image_embeds # torch.multiply(image_prefer, all_h_image_embeds)  # Eq.15 "+"right
        final_h_text_embeds = all_h_text_embeds # torch.multiply(text_prefer, all_h_text_embeds)
        final_hyper_embeds = (1 / 3) * common_embeds + (1 / 3) * final_h_image_embeds + (1 / 3) * final_h_text_embeds # Emul
        final_hyper_embeds = self.alpha * F.normalize(final_hyper_embeds)
        final_h_user_embeds, final_h_item_embeds = torch.split(final_hyper_embeds, [self.n_users, self.n_items], dim=0)

        if self.agg == "concat":
            all_embeddings_items = torch.cat(
                [final_id_emb, final_image_emb, final_text_emb, final_h_item_embeds], dim=1  # fused item
            )  # [items, feat_embed_dim * 2]
            all_embeddings_users = torch.cat(
                [final_id_preference, final_image_preference, final_text_preference, final_h_user_embeds], dim=1 
            )  # [users, feat_embed_dim * 2]
            all_embeddings_items = self.transform(all_embeddings_items)
            all_embeddings_users = self.transform(all_embeddings_users)
        elif self.agg == "sum":
            all_embeddings_items = final_id_emb + final_image_emb + final_text_emb + final_h_item_embeds # [items, feat_embed_dim]
            all_embeddings_users = (
                    final_id_preference + final_image_preference + final_text_preference + final_h_user_embeds
            )  # [users, feat_embed_dim]

        if train:
            return all_embeddings_users, all_embeddings_items, final_id_preference, final_id_emb, final_image_preference, final_text_preference, \
                   final_image_emb, final_text_emb, hyper_image_user_embeds, hyper_text_user_embeds, \
                   hyper_image_item_embeds, hyper_text_item_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        self.forward_adj = self.masked_adj

        ua_embeddings, ia_embeddings, final_id_preference, final_id_emb, final_image_preference, final_text_preference, \
        final_image_emb, final_text_emb, hyper_image_user_embeds, hyper_text_user_embeds, \
        hyper_image_item_embeds, hyper_text_item_embeds = self.forward(self.forward_adj, train=True)
        # hyper_image_item_embeds, hyper_text_item_embeds = self.forward(self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
  
        if self.cl_loss_type == "InfoNCE":
            cl_loss = self.InfoNCE(final_image_preference[users], u_g_embeddings, self.cl_tau) + self.InfoNCE(
                final_text_preference[users], u_g_embeddings, self.cl_tau)
        elif self.cl_loss_type == "batch":
            cl_loss = self.batched_contrastive_loss(final_image_preference[users], u_g_embeddings) + self.batched_contrastive_loss(
                final_text_preference[users], u_g_embeddings)
            cl_loss = self.cl_loss * cl_loss
            if self.cl_loss2 != 0.0:
                local_user = final_id_preference+final_image_preference+final_text_preference
                cl_loss2 = self.batched_contrastive_loss(hyper_image_user_embeds[users], local_user[users]) + self.batched_contrastive_loss(
                hyper_text_user_embeds[users], local_user[users])
                cl_loss2 = self.cl_loss2 * cl_loss2
            else:
                cl_loss2 = 0.0
            cl_loss += cl_loss2
        if self.cl_loss_type == "hcl":
            cl_loss = self.batched_contrastive_loss(final_image_preference[users], u_g_embeddings) + self.batched_contrastive_loss(
                final_text_preference[users], u_g_embeddings)
            cl_loss += self.ssl_triple_loss(hyper_image_user_embeds[users], hyper_text_user_embeds[users],
                                           u_g_embeddings) + self.ssl_triple_loss(hyper_image_item_embeds[pos_items],
                                                                                  hyper_text_item_embeds[pos_items],
                                                                                  pos_i_g_embeddings)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + cl_loss 

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def denoise(self, origin_emb, target_emb):
        """
        将输入的原始嵌入分解为清晰嵌入和噪声嵌入（PyTorch 实现）。
    
        Args:
            origin_emb (torch.Tensor): 原始嵌入，形状为 [batch_size, embedding_dim]
            target_emb (torch.Tensor): 目标嵌入，形状为 [batch_size, embedding_dim]
    
        Returns:
            torch.Tensor: 清晰嵌入，形状为 [batch_size, embedding_dim]
            torch.Tensor: 噪声嵌入，形状为 [batch_size, embedding_dim]
        """
        # 计算内积 res_array
        res_array = torch.sum(origin_emb * target_emb, dim=1, keepdim=True) * target_emb  # [batch_size, embedding_dim]
        
        # 计算归一化因子 norm_num
        norm_num = torch.norm(target_emb, dim=1, keepdim=True) ** 2 + 1e-12  # [batch_size, 1]
        
        # 计算清晰嵌入 clear_emb
        clear_emb = res_array / norm_num  # [batch_size, embedding_dim]
        
        # 计算噪声嵌入 noise_emb
        noise_emb = origin_emb - clear_emb  # [batch_size, embedding_dim]
        
        # 返回缩放后的清晰嵌入和噪声嵌入
        return clear_emb * 0.3, noise_emb * 0.3

