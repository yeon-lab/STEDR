import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

def PEHE(t, y, TE, y0_pred, y1_pred):
    TE_pred = (y1_pred - y0_pred)  
    return torch.mean((TE - TE_pred)**2).item()

def ATE(t, y, TE, y0_pred, y1_pred):
    TE = torch.mean(TE)  
    TE_pred = torch.mean(y1_pred - y0_pred)
    return torch.abs(TE-TE_pred).item()

def RMSE(t, y, TE, y0_pred, y1_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    return torch.sqrt(torch.mean((y - y_pred)**2)).item()

def compute_variances(te, assigned_clusters, n_clusters):
    within_var, across_var = [], []
    for idx in range(n_clusters):
        te_by_cluster = te[assigned_clusters == idx]
        if len(te_by_cluster)>0:
            within_var.append(torch.var(te_by_cluster).item())
            across_var.append(torch.mean(te_by_cluster).item())        
    within_var = np.mean(within_var)
    across_var = np.var(across_var)
    return within_var, across_var

def PEHE(t, y, TE, y0_pred, y1_pred):
    TE_pred = (y1_pred - y0_pred)  
    return np.mean((TE - TE_pred)**2).item()

def ATE(t, y, TE, y0_pred, y1_pred):
    TE = np.mean(TE)  
    TE_pred = np.mean(y1_pred - y0_pred)
    return np.abs(TE-TE_pred).item()

def RMSE(t, y, TE, y0_pred, y1_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    return np.sqrt(np.mean((y - y_pred)**2)).item()

def IPTW(t, y, y0_pred, y1_pred, t_pred):
    iptw = np.mean(((y * t)/t_pred) - ((y * (1-t))/(1-t_pred)))
    return iptw.item()

def Acc_treatment(t, y, y0_pred, y1_pred, t_pred):
    t_pred = np.where(t_pred > 0.5, 1.0, 0.0)
    return balanced_accuracy_score(t, t_pred)

def Acc_outcome(t, y, y0_pred, y1_pred, t_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    y_pred = np.where(y_pred > 0.5, 1.0, 0.0)
    return balanced_accuracy_score(y, y_pred)
    
def AUROC(t, y, y0_pred, y1_pred, t_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    y_pred = np.where(y_pred > 0.5, 1.0, 0.0)
    return roc_auc_score(y, y_pred)    
