import numpy as np
import torch
from trainer.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model.utils import padding
from model.metric import compute_variances

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      metric_ftns,
                      config,
                      train_set,
                      valid_set,
                      test_set):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.maxlen = config['data_loader]['maxlen']
        self.is_EHR = config['data_loader']['data'] == 'EHR'

        self.n_groups = self.model.n_groups
        self.input_dim = self.model.input_dim

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.n_train = len(train_set['X'])
        
        self.train_n_batches = int(np.ceil(float(self.n_train) / float(self.batch_size)))
        self.valid_n_batches = int(np.ceil(float(len(valid_set['X'])) / float(self.batch_size)))
        self.test_n_batches = int(np.ceil(float(len(test_set['X'])) / float(self.batch_size)))

        self.do_validation = self.valid_set is not None
        self.lr_scheduler = optimizer
        self.log_step = 4  # reduce this if you want more logs
        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])


    def _train_epoch(self, epoch):
        self.model.train()
        self.metrics.reset()        
        y0_outs, y1_outs, t_outs = np.array([]), np.array([]), np.array([])
        y_trgs, t_trgs, te_trgs = np.array([]), np.array([]), np.array([])
        assigned_groups = np.array([])
        x_lengths, x_dates = None, None
        for index in range(self.train_n_batches):
            x = self.train_set['X'][index*self.batch_size:(index+1)*self.batch_size]
            if self.is_EHR:
                x_dates = self.train_set['X_dates'][index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = padding(x, self.input_dim, self.maxlen)
            else:
                te = torch.Tensor(
                        self.train_set['TE'][index*self.batch_size:(index+1)*self.batch_size]                    
                    ).to(self.device)
            x = torch.from_numpy(x).float().to(self.device)
            t = torch.Tensor(
                    self.train_set['T'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)            
            y = torch.Tensor(
                    self.train_set['Y'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)

            self.optimizer.zero_grad()
            loss, y0_pred, y1_pred, t_pred, groups = self.model.predict(x, t, y, x_lengths, x_dates)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.metrics.update('loss', loss.item())
            
            if epoch % self.log_step == 0 and index % self.log_step == 0:
                print('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(index),
                    loss.item(),
                ))
                
            assigned_groups = np.append(assigned_groups, groups.cpu().numpy())
            
            y0_outs = np.append(y0_outs, y0_pred.detach().cpu().numpy())
            y1_outs = np.append(y1_outs, y1_pred.detach().cpu().numpy())
            if self.is_EHR:
                t_outs = np.append(t_outs, t_pred.detach().cpu().numpy())
            else:
                te_trgs = np.append(te_trgs, te.cpu().numpy())
            y_trgs = np.append(y_trgs, y.cpu().numpy())
            t_trgs = np.append(t_trgs, t.cpu().numpy())
            
        for met in self.metric_ftns: 
            if self.is_EHR:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, y0_outs, y1_outs, t_outs))
            else:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
                
        log = self.metrics.result() 
        
        if self.is_EHR:
            TE = y1_outs - y0_outs
        else:
            TE = te_trgs

        within_var, across_var = compute_variances(TE, assigned_groups, self.n_groups)
        group_list, counts = np.unique(assigned_groups, return_counts=True)
        
        log.update({'clusters': group_list.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'within_var':within_var})
        log.update({'across_var':across_var})  
        
        for c in group_list.tolist():
            sub_te = TE[assigned_groups==c]
            log.update({'TE avg. for group {}'.format(c): np.mean(sub_te)})
            log.update({'TE std. for group {}'.format(c): np.std(sub_te)})
        
        if self.do_validation:
            valid_results = self._infer(self.valid_set, self.valid_n_batches, is_val=True)
            log.update(**{'val_' + k: v for k, v in valid_results['log'].items()})
            
        return log

        
    def _infer(self, data_set, n_batches, is_val=False ,phase=False):
        self.metrics.reset()      
        self.model.train()
        y0_outs, y1_outs, t_outs = np.array([]), np.array([]), np.array([])
        y_trgs, t_trgs, te_trgs = np.array([]), np.array([]), np.array([])
        assigned_groups  = np.array([])
        x_lengths, x_dates = None, None
        
        for index in range(n_batches):
            x = data_set['X'][index*self.batch_size:(index+1)*self.batch_size]
            if self.is_EHR:
                x_dates = data_set['X_dates'][index*self.batch_size:(index+1)*self.batch_size]
                x, x_lengths = padding(x, self.input_dim, self.maxlen)
            else:
                te = torch.Tensor(
                        data_set['TE'][index*self.batch_size:(index+1)*self.batch_size]                    
                    ).to(self.device)
            x = torch.from_numpy(x).float().to(self.device)
            t = torch.Tensor(
                    data_set['T'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)            
            y = torch.Tensor(
                    data_set['Y'][index*self.batch_size:(index+1)*self.batch_size]                    
                ).to(self.device)
            
            loss, y0_pred, y1_pred, t_pred, groups = self.model.predict(x, t, y, x_lengths, x_dates)
            self.metrics.update('loss', loss.item())

            assigned_groups = np.append(assigned_groups, groups.cpu().numpy())
            
            y0_outs = np.append(y0_outs, y0_pred.detach().cpu().numpy())
            y1_outs = np.append(y1_outs, y1_pred.detach().cpu().numpy())
            if self.is_EHR:
                t_outs = np.append(t_outs, t_pred.detach().cpu().numpy())
            else:
                te_trgs = np.append(te_trgs, te.cpu().numpy())
            y_trgs = np.append(y_trgs, y.cpu().numpy())
            t_trgs = np.append(t_trgs, t.cpu().numpy())
            
        for met in self.metric_ftns: 
            if self.is_EHR:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, y0_outs, y1_outs, t_outs))
            else:
                self.metrics.update(met.__name__, met(t_trgs, y_trgs, te_trgs, y0_outs, y1_outs))
                
        log = self.metrics.result() 
        
        if self.is_EHR:
            TE = y1_outs - y0_outs
        else:
            TE = te_trgs
        within_var, across_var = compute_variances(TE, assigned_groups, self.n_groups)
        group_list, counts = np.unique(assigned_groups, return_counts=True)
        
        log.update({'clusters': group_list.tolist()})
        log.update({'counts': counts.tolist()})
        log.update({'within_var':within_var})
        log.update({'across_var':across_var})  
        
        for c in group_list.tolist():
            sub_te = TE[assigned_groups==c]
            log.update({'TE avg. for group {}'.format(c): np.mean(sub_te)})
            log.update({'TE std. for group {}'.format(c): np.std(sub_te)})
            
        return log
        

    def _test_epoch(self):
        log = {}        
        train_log = self._infer(self.train_set, self.train_n_batches, phase='train')
        valid_log = self._infer(self.valid_set, self.valid_n_batches, phase='valid')
        test_log = self._infer(self.test_set, self.test_n_batches, phase='test')
        
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
                        
        print('='*100)
        print('Inference is completed')
        print('-'*100)
        for key, value in log.items():
            print('    {:20s}: {}'.format(str(key), value))  
        print('='*100)
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        return base.format(current, self.n_train, 100.0 * current / self.n_train)
        
        
