import numpy as np 
from tqdm import tqdm 
import os, logging, sys  
import torch 
from torch.distributions import Normal 
import time 
from .train import trainer 
from ..model.metrics import metrics 
from pathlib import Path 
import warnings 
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__) 

class trainerProb(trainer): 
    """
    Parent class for training 
    """
    def __init__(self, model, device, args, dirs, hand_data): 
        super(trainerProb, self).__init__(model, device, args, dirs, hand_data)

    def _loss_fn(self, loc, variance, y, pi=None):
        """
        Args:
            loc: [batch_size] estimated mean at time step t 
            variance: [batch_size] estimated standard deviation at time step t 
            y: [batch_size] z_t
        Returns:
            loss_batch: (Variable) average log-likelihood loss across the batch 
        """
        if self.args.pdf == 'gaussian':
            scale = variance.sqrt() 
            distr = Normal(loc, scale) 
        else:
            raise Warning('Not supported loss function') 
        loss_batch = -distr.log_prob(y).mean() 
        return loss_batch, scale, distr   

    def _train_epoch(self, id_epoch, train_dl, optimizer, criterion, mode): 
        if mode == 'train':         # Training
            self.model.train() 
            tqdm_string = 'Train Epoch'
            epochs = self.args.epochs 
        elif mode == 'valid':       # Validing 
            self.model.eval() 
            tqdm_string = 'Valid Epoch' 
            epochs = self.args.epochs 
        elif mode == 'test':        # Testing 
            self.model.eval() 
            tqdm_string = 'Test Epoch'
            epochs = 1 
        else:
            raise Warning('Not supported traing mode') 
        
        num_samples = len(train_dl.dataset) 
        # Quantile forecasting 
        quantiles = torch.arange(1, 10, 2, dtype=torch.float32, device=self.device) / 10 
        num_forecasts = len(quantiles) 
        # Initiate dimensions and book-keeping variables 
        # (num_forecasts, n_seq, num_samples, dim_output)
        yhat_tot = np.zeros((num_forecasts, self.args.n_seq, num_samples, self.hand_data.dim_output), dtype='float32') 
        # (n_seq, num_samples, dim_output)
        y_tot = np.zeros((self.args.n_seq, num_samples, self.hand_data.dim_output), dtype='float32') 
        # (window, num_samples, dim_input)
        x_tot = np.zeros((self.hand_data.window, num_samples, self.hand_data.dim_input), dtype='float32') 
        train_mse = 0 
        
        n_samples_dist = 1000 
        # loop 
        sigma = []
        with tqdm(total=num_samples, ncols=150, 
            desc=f"[{tqdm_string} {id_epoch+1:3d}/{epochs}/{self.args.detect_cycle_id}/{self.args.detect_cycle}]") as pbar_train: 
            for mbidx, batch in enumerate(train_dl): 
                # Batch 
                j = np.min(((mbidx + 1) * self.args.batch_size, num_samples))
                # obtain input and target data | Send to device 
                x, y = batch[0].to(self.device), batch[1].to(self.device) 
                # Permute to [seqlen x batch x feature] 
                x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)       # (n_lag+n_seq, batch_size, dim_input) (n_seq, batch_size, dim_output)
                # Fill bookkeeping variables 
                y_tot[:, mbidx*self.args.batch_size:j] = y.cpu().detach().numpy()                           # (n_seq, num_samples, dim_output)   
                x_tot[:, mbidx*self.args.batch_size:j] = x[:self.hand_data.window].cpu().detach().numpy()   # (window, num_samples, dim_input)  
                
                # Create lags and covariate tensors 
                if self.args.scaling_train: 
                    scaleY = 1 + x[:self.args.n_lag, :, -self.hand_data.d_lag:].mean(dim = 0)               # (batch_size, 1) 
                    x[:, :, -self.hand_data.d_lag:] /= scaleY                                               # (n_lag+n_seq, batch_size, 1)
                    y /= scaleY                                                                             # (n_seq, batch_size, 1)
                else:
                    scaleY = torch.tensor([1.0]) 
                # Create three inputs: (i) lags, (ii) time series index, (iii) covariates
                # Series  MicrosecondOfSecond  SecondOfMinute  MinuteOfHour Wave_elevation_lag
                X_idx = x[:, :, 0:self.hand_data.d_emb].long()                                              # (n_lag+n_seq, batch_size, 1) 
                X_cov = x[:, :, self.hand_data.d_emb:self.hand_data.d_emb+self.hand_data.d_cov]             # (n_lag+n_seq, batch_size, 3) 
                X_lag = x[:self.hand_data.window, :, -self.hand_data.d_lag:]                                # (n_lag+n_seq, batch_size, 1) 
                if mode == 'train':   # Training 
                    # Set gradients to zero of optimizer 
                    optimizer.zero_grad() 
                    # Calculate loc and scale parameters of output distribution 
                    mean, variance = self.model(X_lag, X_cov, X_idx, self.args.n_seq) 
                    # loss 
                    loss_batch, scale, distr = self._loss_fn(mean, variance, y) 
                    sigma.append((scale*scaleY).permute(1, 0, 2).cpu().detach().clone()) 
                    # Backward pass 
                    loss_batch.backward() 
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.1)    
                    # Update parameters 
                    optimizer.step() 
                elif mode == 'valid' or mode == 'test':       # Validing or Testing 
                    with torch.no_grad(): 
                        mean_prev = X_lag[self.args.n_lag, :, [-1]].clone().detach()    # (n_lag+n_seq, batch_size, 1)  
                        for t in range(self.args.n_seq): 
                            X_lag[self.args.n_lag+t, :, [-1]] = mean_prev 
                            mean, variance = self.model(X_lag[:self.args.n_lag + t + 1], X_cov, X_idx, t + 1) 
                            mean_prev = mean[-1].clone().detach()  
                        # Calculate loss 
                        loss_batch, scale, distr = self._loss_fn(mean, variance, y) 
                        sigma.append((scale*scaleY).permute(1, 0, 2).cpu().detach().clone())
                else:
                    raise Warning('Not supported traing mode') 

                # calculate accumulate error 
                train_mse += loss_batch.item() 
                yhat = distr.sample([n_samples_dist])                   # (n_samples_dist, n_seq, batch_size, dim_output)
                yhat *= scaleY                                          # (n_samples_dist, n_seq, batch_size, dim_output) 
                yhat_q = torch.quantile(yhat, quantiles, dim=0)         # (num_forecasts, n_seq, batch_size, dim_output)
                yhat_tot[:, :, mbidx*self.args.batch_size:j, :] = yhat_q.detach().cpu().numpy()     # (num_forecasts, n_seq, num_samples, dim_output)
                pbar_train.set_postfix({'loss': '{0:1.5f}'.format(train_mse/(mbidx+1))}) 
                pbar_train.update(x.shape[1]) 

            # Calculate relative accuracy # Cat and (torch 2 numpy) 
            sigma = torch.cat(sigma, dim=0).numpy()                     # shape (batch_size, seq_len, dim)
            y_test, y_pred = [], []
            for i in range(y_tot.shape[1]):
                y_test.append(y_tot[:, i, :])
                y_pred.append(yhat_tot[2, :, i, :])
            y_test = np.array(y_test)                                   # shape (batch_size, seq_len, dim)
            y_pred = np.array(y_pred)                                   # shape (batch_size, seq_len, dim) 
            
            output = 0 
            y, yhat = y_tot[:, :, output], yhat_tot[:, :, :, output]    # (seqlen_output, num_samples) (num_forecasts, seqlen_output, num_samples)
            df, Rp_result, ACC, dist_acc, nrmse, ndrmse = metrics(yhat, y, quantiles.cpu().numpy(), mode=mode, args=self.args) 
            if self.args.eval_standard == 'prob_acc':
                pbar_train.set_postfix({'loss': '{0:1.5f}'.format(train_mse/(mbidx+1)), 'Rp': '{0:1.5f}'.format(Rp_result), 
                    'prob_acc': '{0:1.3f}'.format(dist_acc), 'ndrmse': '{0:1.3f}'.format(ndrmse)}) 
            elif self.args.eval_standard == 'loss' or self.args.eval_standard == 'ndrmse':
                pbar_train.set_postfix({'loss': '{0:1.5f}'.format(train_mse/(mbidx+1)), 'Rp': '{0:1.5f}'.format(Rp_result), 
                    'ndrmse': '{0:1.3f}'.format(ndrmse)}) 
            else:
                logger.info(f"The self.args.eval_standard {self.args.eval_standard} is not defined !") 
                os.system("pause")
                sys.exit(1)

        if mode == 'train' or mode == 'valid':
            return train_mse/len(train_dl), ACC, dist_acc, Rp_result, nrmse, ndrmse   
        else:
            return x_tot, y_tot, yhat_tot, ACC, dist_acc, Rp_result, nrmse, ndrmse, df, sigma   

    def train(self, epochs, optimizer, scheduler, criterion, training_loader, validing_loader):
        logger.info(f"Begaining Training.") 
        if torch.cuda.is_available():
            torch.cuda.synchronize() 
        start_time = time.perf_counter()

        t_epochs = np.zeros((epochs)) 
        train_loss, train_acc, train_dist_acc, train_Rp, train_nrmse, train_ndrmse = \
            np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs) 
        valid_loss, valid_acc, valid_dist_acc, valid_Rp, valid_nrmse, valid_ndrmse = \
            np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)  

        self.valid_min, self.stop_metric = np.inf, np.inf  
        self.valid_max = 0 
        self.stop_round = 0 
        for epoch in range(epochs): 
            # Training 
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            epoch_start = time.perf_counter()
            train_loss[epoch], train_acc[epoch], train_dist_acc[epoch], \
                train_Rp[epoch], train_nrmse[epoch], train_ndrmse[epoch]  = \
                self._train_epoch(id_epoch=epoch, train_dl=training_loader,optimizer=optimizer, 
                    criterion=criterion, mode='train')

            # Evaluation 
            valid_loss[epoch], valid_acc[epoch], valid_dist_acc[epoch], \
                valid_Rp[epoch], valid_nrmse[epoch], valid_ndrmse[epoch] = \
                self._train_epoch(id_epoch=epoch, train_dl=validing_loader,optimizer=optimizer, 
                    criterion=criterion, mode='valid')

            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            epoch_end = time.perf_counter() 
            t_epochs[epoch] = epoch_end - epoch_start 

            if not self.args.detect_hardware:
                if self.args.eval_standard == 'loss':
                    acc_flag = self._save_best_models(metrics=valid_loss[epoch], optimizer=optimizer, epoch=epoch)
                elif self.args.eval_standard == 'ndrmse':
                    acc_flag = self._save_best_models(metrics=valid_ndrmse[epoch], optimizer=optimizer, epoch=epoch) 
                elif self.args.eval_standard == 'prob_acc':
                    acc_flag = self._save_best_models(metrics=valid_dist_acc[epoch], optimizer=optimizer, epoch=epoch) 
                else:
                    raise Warning('Not supported save best model standard') 

                # Early Stop Learning 
                self._early_stop2(stop_flag = acc_flag)
                if self.stop_round >= self.args.early_stop:
                    logger.info(f'Early STOP')
                    break 
            
                if self.args.scheduler_dynamic:
                    # Progress learning rate scheduler
                    if self.args.optimizer_mode == 1:
                        if self.args.eval_standard == 'loss':
                            scheduler.step(metrics=valid_loss[epoch]) 
                        elif self.args.eval_standard == 'ndrmse': 
                            scheduler.step(metrics=valid_ndrmse[epoch]) 
                        elif self.args.eval_standard == 'prob_acc': 
                            scheduler.step(metrics=valid_dist_acc[epoch]) 
                        else:
                            raise Warning('Not supported save best model standard Line 240') 
                    elif self.args.optimizer_mode == 2:
                        scheduler.step() 

                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break 

                if (epoch+1) % 1 == 0:
                    logger.info(f"The {epoch+1} epoch learning rate is {cur_lr}") 

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        logger.info(f"The total epochs training time is {end_time - start_time:.5f} s") 
        logger.info(f"The average epoch training time is {t_epochs[:epoch+1].mean():.5f} s")

        return train_loss[:epoch+1], valid_loss[:epoch+1], train_Rp[:epoch+1], valid_Rp[:epoch+1],\
            train_acc[:epoch+1], valid_acc[:epoch+1], train_dist_acc[:epoch+1], valid_dist_acc[:epoch+1], \
            train_nrmse[:epoch+1], valid_nrmse[:epoch+1], train_ndrmse[:epoch+1], valid_ndrmse[:epoch+1], \
            epoch+1, end_time - start_time, t_epochs[:epoch+1].mean()

    def test(self, testing_loader): 
        # Testing 
        # (window, num_samples, dim_input) (n_seq, num_samples, dim_output) (num_forecasts, n_seq, num_samples, dim_output) 
        if self.args.keep_best:
            keep_acc, keep_dist_acc = 0, 0  
            keep_nrmse, keep_ndrmse, keep_rp = np.inf, np.inf, np.inf   
            keep_model = None 
            for modelpath, score in self.valid_keep.best_scores.items():
                logger.info(f"The Valid score is {score:.5f}")
                self.model.load_model(name=self.filename, file_or_path_directory=modelpath) 
                if torch.cuda.is_available(): 
                    torch.cuda.synchronize() 
                inference_start = time.perf_counter()
                x_tot_, y_tot_, yhat_tot_, mean_acc_, dist_acc_, Rp_result_, nrmse_, ndrmse_, df_, sigma_ = self._train_epoch(
                    id_epoch=0, train_dl=testing_loader, optimizer=None, criterion=None, mode='test') 
                if torch.cuda.is_available():
                    torch.cuda.synchronize() 
                inference_end = time.perf_counter() 
                logger.info(f"The inference time is {inference_end - inference_start:.5f} s") 
                keep_flag = False 
                if self.args.eval_standard == 'prob_acc':
                    if dist_acc_ > keep_dist_acc:
                        keep_dist_acc = dist_acc_ 
                        keep_flag = True 
                elif self.args.eval_standard == 'ndrmse':
                    if ndrmse_ < keep_ndrmse:
                        keep_ndrmse = ndrmse_
                        keep_flag = True 
                elif self.args.eval_standard == 'loss': 
                    if Rp_result_ < keep_rp:
                        keep_rp = Rp_result_ 
                        keep_flag = True 
                else:
                    logger.info(f"The self.args.eval_standard {self.args.eval_standard} is not defined !") 
                    os.system("pause")
                    sys.exit(1)
                if keep_flag:
                    keep_model = modelpath 
                    x_tot, y_tot, yhat_tot, mean_acc, dist_acc, Rp_result, nrmse, ndrmse, df, sigma = \
                        x_tot_, y_tot_, yhat_tot_, mean_acc_, dist_acc_, Rp_result_, nrmse_, ndrmse_, df_, sigma_ 

            directory = Path(self.filepath)
            checkpoints = list(directory.glob('%s*' % self.filename)) 
            for checkpoint in checkpoints:
                if checkpoint != keep_model:
                    os.remove(checkpoint)
        else:
            self.model.load_model(name=self.filename, file_or_path_directory=self.filepath) 
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            inference_start = time.perf_counter()
            x_tot, y_tot, yhat_tot, mean_acc, dist_acc, Rp_result, nrmse, ndrmse, df, sigma = self._train_epoch(
                id_epoch=0, train_dl=testing_loader, optimizer=None, criterion=None, mode='test') 
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            inference_end = time.perf_counter() 
            logger.info(f"The inference time is {inference_end - inference_start:.5f} s") 

        sc_test_x, sc_test_y, sc_pred_y = [], [], []
        sc_pred_y_1, sc_pred_y_9 = [], []
        for i in range(x_tot.shape[1]):
            sc_test_x.append(x_tot[:self.args.n_lag+1, i, -self.hand_data.d_lag:]) 
            sc_test_y.append(y_tot[:, i, :])
            sc_pred_y.append(yhat_tot[2][:, i, :])
            sc_pred_y_1.append(yhat_tot[0][:, i, :])
            sc_pred_y_9.append(yhat_tot[4][:, i, :])
        
        sc_test_x = np.array(sc_test_x)        # (num_samples, n_lag+1)
        sc_test_y = np.array(sc_test_y)        # (num_samples, n_seq)
        sc_pred_y = np.array(sc_pred_y)        # (num_samples, n_seq) 5 
        sc_pred_y_1 = np.array(sc_pred_y_1)     # 1 
        sc_pred_y_9 = np.array(sc_pred_y_9)     # 9 

        sc_test_y_add = []
        for i in range(sc_test_y.shape[0]):
            sc_test_y_add.append(np.append(sc_test_x[i, -1], sc_test_y[i, :])) 
        sc_test_y_add = np.array(sc_test_y_add).reshape(-1, self.args.n_seq+1, 1)
        
        return sc_test_x, sc_test_y_add, sc_pred_y, sc_pred_y_1, sc_pred_y_9, \
            mean_acc, dist_acc, df, sigma, Rp_result, nrmse, ndrmse     