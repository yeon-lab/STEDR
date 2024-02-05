from torch.nn import Parameter
from typing import Optional
from torch.autograd import Variable
import torch.nn.functional as F
from functools import partial
from copy import deepcopy

def loss_func(y, t, y0_pred, y1_pred, t_pred):
    loss_t = F.binary_cross_entropy(t_pred, t)
    loss_y = torch.sum((1. - t) * torch.square(y - y0_pred)) + torch.sum(t * torch.square(y - y1_pred))
    return loss_y + loss_t


class STEDR(nn.Module):
    def __init__(self, config, alpha=0.1):
        super(STEDR, self).__init__()
        
        self.input_dim = config['input_dim']
        att_dim = config['att_dim']
        emb_dim = config['emb_dim']
        dist_dim = config['dist_dim']
        n_layer = config['n_layers']
        maxlen = config['maxlen'] # maximum visits
        self.n_clusters = config['n_clusters']
        
        nhead = 1
        for i in range(2, self.input_dim):
            if self.input_dim % i == 0:
                nhead = i
                break

        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.CE = nn.CrossEntropyLoss()
        self.KLD = nn.KLDivLoss(reduction="batchmean")
        self.criterion = partial(loss_func)        
        
        self.attention_layers = nn.ModuleList([nn.Linear(maxlen, att_dim) for _ in range(self.input_dim)])
        self.attention_weights = nn.Parameter(torch.rand(att_dim, 1))
        
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.ReLU()
            )
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=nhead, batch_first=True, dim_feedforward=emb_dim) 
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)
        
        self.decoder = nn.Sequential(
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, self.input_dim)
            )
    
        self.global_fc_mu = nn.Linear(self.input_dim, dist_dim)
        self.global_fc_var = nn.Linear(self.input_dim, dist_dim)
        
        self.cluster_fc_mu, self.cluster_fc_var=nn.ModuleList([]), nn.ModuleList([])
        for i in range(self.n_clusters):
            self.cluster_fc_mu.append(nn.Linear(self.input_dim, dist_dim))
            self.cluster_fc_var.append(nn.Linear(self.input_dim, dist_dim))

        outcomemodel = nn.Sequential(
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, 1)
            )
        
        self.control_out = deepcopy(outcomemodel)
        self.treat_out = deepcopy(outcomemodel)
        
        self.cluster_out = nn.Sequential(
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, self.n_clusters), nn.Softmax(dim=1)
            )
        self.propensity = nn.Sequential(
            nn.Linear(dist_dim, dist_dim), nn.ReLU(),
            nn.Linear(dist_dim, 1), nn.Sigmoid()
            )

    def target_distribution(self, q):
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p + 1e-5
    
    def similarity(self, query, keys):    
        norm_squared = torch.sum((query.unsqueeze(1) - keys) ** 2, -1)
        numerator = 1.0 / (1.0 + (norm_squared / self.tau))
        power = float(self.tau + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std 
    
    def reparameterize_gmm(self, cluster_mu, cluster_var, weights):
        mixtured_features = 0
        for i in range(self.n_clusters):
            std = torch.exp(0.5 * cluster_var[:,i])
            eps = torch.randn_like(std)
            mixtured_features += weights[:,i].unsqueeze(-1)*(cluster_mu[:,i] + std) 
        return mixtured_features
    
    def kl_div_dist(self, local_, global_):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        local_ = F.log_softmax(local_, dim=1)
        global_ = F.softmax(global_, dim=1)
        return kl_loss(local_, global_)
    
    def global_features(self, x):
        global_mu = self.global_fc_mu(x)
        global_var = self.global_fc_var(x)
        global_z = self.reparameterize(global_mu, global_var)
        return global_z, global_mu, global_var
    
    def local_features(self, x):
        cluster_z, cluster_mu, cluster_var = torch.tensor([]).to(x.device), torch.tensor([]).to(x.device), torch.tensor([]).to(x.device)
        for i in range(self.n_clusters):
            mu = self.cluster_fc_mu[i](x)
            logvar = self.cluster_fc_var[i](x)
            z = self.reparameterize(mu, logvar)
            cluster_z = torch.cat([cluster_z, z.unsqueeze(1)], dim=1)
            cluster_mu = torch.cat([cluster_mu, mu.unsqueeze(1)], dim=1)
            cluster_var = torch.cat([cluster_var, logvar.unsqueeze(1)], dim=1)
        return cluster_z, cluster_mu, cluster_var

    def encode_input(self, x): # get attentions for variable-level
        n_samples = len(x)
        attentions = torch.tensor([])
        embedded_data = [self.attention_layers[i](x[:, :, i].view(n_samples, -1)) for i in range(self.input_dim)]
        embedded_data = torch.stack(embedded_data, dim=1)
        attentions = torch.matmul(embedded_data, self.attention_weights).squeeze(-1)
        attentions = torch.softmax(attentions, dim=1).unsqueeze(1) 
        attended_input = attentions * x
        attended_input = self.embedding(attended_input)
        x_encoded = self.encoder(attended_input)
        return x_encoded, attentions.squeeze(1) 
    
    def get_features(self, x, x_lengths=None):
        n_samples = len(x)
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        if x_lengths is not None:
            index = x_lengths-1
        else:
            index = -1
            
        x_encoded, attentions = self.encode_input(x) # attened input
        global_z, global_mu, global_var = self.global_features(x_encoded) # get global dist that can represent all dat
        cluster_z, cluster_mu, cluster_var = self.local_features(x_encoded) # get local dist for each cluster
            
        global_z_ = global_z[np.arange(n_samples),index]
        cluster_z_ = cluster_z[np.arange(n_samples),:,index]
        
        weights = self.similarity(global_z_, cluster_z_)  # compute all cluster simiarities to global dist
        clusters = torch.argmax(weights, dim=1) # find cluster for each data with highest simiarity to global dist
        features = cluster_z_[torch.arange(n_samples), clusters] # get local feature according to assigned cluster to each data

        mse_loss = self.reconstruction_loss(global_z, x)
        p = self.target_distribution(weights) # set target distribution
        target_dist_loss = (weights*torch.log(p)).sum()
        mixtured_z = self.reparameterize_gmm(cluster_mu[np.arange(n_samples),:,index], 
                                             cluster_var[np.arange(n_samples),:,index], 
                                             weights) # get GMM using similarities and local dist variables
        kl_loss = self.kl_div_dist(mixtured_z, global_z_) # KL divergence between GMM and global
        
        loss = mse_loss + kl_loss + target_dist_loss
        
        return loss, features, clusters, attentions, global_z_
      
    
    def predict(self, x, x_dates, t, y, lengths=None):
        loss, features, clusters, attentions, global_z_ = self.get_features(x, lengths) # get features and clusters       
        y0_pred = self.control_out(features) # predict control outcome
        y1_pred = self.treat_out(features) # predict treated outcome
        t_pred = self.propensity(features) # predict propensity score
        te_pred = y1_pred-y0_pred
        
        pred_loss = self.criterion(y, t, y0_pred, y1_pred, t_pred) 
        
        c_pred = self.cluster_out(features) # predict propensity score
        c_loss = self.CE(c_pred, clusters)
        c_pred = torch.argmax(c_pred, dim=1)
        
        CIs = []
        within_var, across_var = torch.tensor([]), torch.tensor([])
        for c in range(self.n_clusters):
            te_sub = te_pred[c_pred==c]
            if len(te_sub) == 0:
                continue    
            lower_bound, upper_bound = torch.quantile(te_sub, torch.tensor([0.025, 0.975]))
            CIs.append([lower_bound, upper_bound])
        overlap = ci_overlap_penalty(CIs)                      
        
        all_loss = pred_loss + loss + c_loss+ self.alpha*(overlap)
        return all_loss, y0_pred, y1_pred, t_pred, clusters, attentions
    
    def reconstruction_loss(self, global_z, x):
        x_hat = self.decoder(global_z) # reconstruct using global dist
        mse_loss = self.MSE(x, x_hat) # reconstruct loss
        return mse_loss

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
def calculate_overlap(ci_a, ci_b):
    lower_bound = torch.max(ci_a[0], ci_b[0])
    upper_bound = torch.min(ci_a[1], ci_b[1])
    overlap = torch.max(upper_bound - lower_bound, torch.tensor(0.0))
    return overlap

def ci_overlap_penalty(subgroup_cis):
    penalty = 0.0
    num_subgroups = len(subgroup_cis)
    for i in range(num_subgroups):
        for j in range(i + 1, num_subgroups):
            overlap = calculate_overlap(subgroup_cis[i], subgroup_cis[j])
            penalty += overlap
    return penalty


    

    
