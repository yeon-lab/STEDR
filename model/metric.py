import numpy as np
from sklearn.metrics import roc_auc_score

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

def AUROC_outcome(t, y, y0_pred, y1_pred, t_pred):
    y_pred = t * y1_pred + (1 - t) * y0_pred
    y_pred = np.where(y_pred > 0.5, 1.0, 0.0)
    return roc_auc_score(y, y_pred)

def AUROC_treatment(t, y, y0_pred, y1_pred, t_pred):
    t_pred = np.where(t_pred > 0.5, 1.0, 0.0)
    return roc_auc_score(t, t_pred)    
