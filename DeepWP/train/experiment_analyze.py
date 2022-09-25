from ..data import Data_Analyze 
import numpy as np 
import pandas as pd 
import os, logging, sys  
import h5py 
from ..data.data_allanalyze import ALLDataAnalyze 

logger = logging.getLogger(__name__) 

class KFlodsAnalyze(object):
    def __init__(self, args, n_rows):
        self.args = args 
        if not args.detect_hardware:
            if args.forecast_type == 'probability': 
                if args.eval_standard == 'prob_acc': 
                    self.columns = ['Model','Epochs', 'Time','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p10','p50','p90','mp50','ACC_70','ACC_80','ACC_90','ACC (%)'] 
                else:
                    self.columns = ['Model','Epochs', 'Time','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p10','p50','p90','mp50'] 
            else:
                logger.info(f"The forecast_type {args.forecast_type} is not defined")
                os.system("pause") 
                sys.exit(1)
        else:
            self.columns = ['Model', 'start_time', 'end_time', 'epoch_time', 'parameters'] 
        self.data = np.zeros((n_rows, len(self.columns))).astype(np.float32) 
        self.result_df = pd.DataFrame(self.data, columns=self.columns) 

    def kflods_train_analyze(self, experiment_analyze):
        # Train analyze 
        self.epochs, self.time = experiment_analyze.train_result_analyze()

    def kflods_test_analyze(self, experiment_analyze, idx):
        self.result_df.loc[idx, 'Model'] = f"{self.args.model_name}_{self.args.covariate_mode}_{self.args.n_lag}_{self.args.n_seq}"
        if self.args.forecast_type == 'probability':
            if self.args.eval_standard == 'prob_acc':
                sMAPE, NRMSE, NDRMSE, ND, RMSE, MAPE, p10, p50, p90, mp50, acc_70, acc_80, acc_90, ACC = \
                    experiment_analyze.test_result_analyze() 
            else:
                sMAPE, NRMSE, NDRMSE, ND, RMSE, MAPE, p10, p50, p90, mp50 = experiment_analyze.test_result_analyze()
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not defined")
            os.system("pause")
            sys.exit(1)
        
        self.result_df.loc[idx, 'Epochs'] = self.epochs  
        self.result_df.loc[idx, 'Time'] = self.time 
        self.result_df.loc[idx, 'sMAPE'] = np.round(sMAPE,3) 
        self.result_df.loc[idx, 'NRMSE'] = np.round(NRMSE,3) 
        self.result_df.loc[idx, 'NDRMSE'] = np.round(NDRMSE,3)
        self.result_df.loc[idx, 'RMSE'] = np.round(RMSE,3)
        self.result_df.loc[idx, 'MAPE'] = np.round(MAPE,3) 
        self.result_df.loc[idx, 'p50'] = np.round(p50,3) 
        if self.args.forecast_type == 'probability':
            self.result_df.loc[idx, 'p10'] = np.round(p10,3) 
            self.result_df.loc[idx, 'p90'] = np.round(p90,3) 
            self.result_df.loc[idx, 'mp50'] = np.round(mp50,3)  
        if self.args.eval_standard == 'prob_acc':
            self.result_df.loc[idx, 'ACC_70'] = np.round(acc_70,3) 
            self.result_df.loc[idx, 'ACC_80'] = np.round(acc_80,3)  
            self.result_df.loc[idx, 'ACC_90'] = np.round(acc_90,3)  
            self.result_df.loc[idx, 'ACC (%)'] = np.round(ACC,3) 

    def load_k_fold_best(self, total_analyze, idx, filepath):
        # 为了避免显存溢出的问题 每个算例均将以往的结果保存一次 
        if self.args.model_configpath is not None: 
            _, config_file = os.path.split(self.args.model_configpath) 
            filesuffix, _ = os.path.splitext(config_file) 
        else:
            curT = datetime.datetime.now() 
            filesuffix = f"{curT.year}_{curT.month}_{curT.day}_{curT.hour}_{curT.minute}_{curT.minute}" 
        debug_file = f"{filesuffix}_src={self.args.data_source}" 
        if not self.args.detect_hardware:
            sigle_filepath = os.path.join(filepath, debug_file+'_overflow.csv') 
        else:
            sigle_filepath = os.path.join(filepath, debug_file+'_hardware.csv') 
        total_analyze.result_df[:(idx+1)].to_csv(sigle_filepath, index=False) 

    def output_k_fold_best(self,filepath):
        analyzeall = ALLDataAnalyze(args=self.args, dir=filepath)
        # result 2 .csv format file 
        analyzeall.result2csv(df=self.result_df)
        # result 2 plot 
        df_values = self.result_df.values 
        n_rows, n_cols = df_values.shape 
        # split 
        p50Epoch = self.result_df['p50'].values 
        if self.args.forecast_type == 'probability':
            p10Epoch = self.result_df['p10'].values 
            p90Epoch = self.result_df['p90'].values 
        ndrmseEpoch = self.result_df['NDRMSE'].values 
        if self.args.eval_standard == 'prob_acc':
            accEpoch = self.result_df['ACC (%)'].values 
        else: 
            accEpoch = self.result_df['NRMSE'].values 
            
        # p50 & acc 
        p50, ndrmse, acc = [], [], [] 
        if self.args.forecast_type == 'probability':
            p10, p90 = [], []
        for i in range(self.args.n_covariate): # 循环遍历变量模式 
            p50_temp = p50Epoch[i:n_rows:self.args.n_covariate].reshape(-1, self.args.n_modeltype).T 
            ndrmse_temp = ndrmseEpoch[i:n_rows:self.args.n_covariate].reshape(-1, self.args.n_modeltype).T 
            acc_temp = accEpoch[i:n_rows:self.args.n_covariate].reshape(-1, self.args.n_modeltype).T 
            p50.append(p50_temp) 
            ndrmse.append(ndrmse_temp)
            acc.append(acc_temp) 
            if self.args.forecast_type == 'probability': 
                p10_temp = p10Epoch[i:n_rows:self.args.n_covariate].reshape(-1, self.args.n_modeltype).T 
                p90_temp = p90Epoch[i:n_rows:self.args.n_covariate].reshape(-1, self.args.n_modeltype).T 
                p10.append(p10_temp) 
                p90.append(p90_temp) 

        p50 = np.array(p50)     # (self.args.n_covariate, self.args.n_modeltype, times) 
        if self.args.forecast_type == 'probability': 
            p10 = np.array(p10) 
            p90 = np.array(p90) 
        ndrmse = np.array(ndrmse)
        acc = np.array(acc)     # (self.args.n_covariate, self.args.n_modeltype, times) 
        # Rp 
        titles = ['Wave Height', 'Position', 'Category', 'Timestamp', 'Global Information']     # length = self.args.n_covariate 
        legends = [f"Hyper_{i+1}" for i in range(self.args.n_modeltype)]
        analyzeall.plot_differnet_covariate(p50, 'rp5', legends, titles, 'avg', self.args.n_seq) 
        analyzeall.plot_differnet_covariate(p50, 'rp5', legends, titles, 'min', self.args.n_seq) 
        if self.args.forecast_type == 'probability': 
            analyzeall.plot_differnet_covariate(p10, 'rp1', legends, titles, 'avg', self.args.n_seq) 
            analyzeall.plot_differnet_covariate(p90, 'rp9', legends, titles, 'avg', self.args.n_seq) 
            analyzeall.plot_differnet_covariate(p10, 'rp1', legends, titles, 'min', self.args.n_seq) 
            analyzeall.plot_differnet_covariate(p90, 'rp9', legends, titles, 'min', self.args.n_seq) 
        analyzeall.plot_differnet_covariate(ndrmse, 'ndrmse', legends, titles, 'avg', self.args.n_seq)
        analyzeall.plot_differnet_covariate(ndrmse, 'ndrmse', legends, titles, 'min', self.args.n_seq)
        if self.args.eval_standard == 'prob_acc':
            analyzeall.plot_differnet_covariate(acc, 'acc', legends, titles, 'avg', self.args.n_seq)
            analyzeall.plot_differnet_covariate(acc, 'acc', legends, titles, 'min', self.args.n_seq)
        else:
            analyzeall.plot_differnet_covariate(acc, 'nrmse', legends, titles, 'avg', self.args.n_seq) 
            analyzeall.plot_differnet_covariate(acc, 'nrmse', legends, titles, 'min', self.args.n_seq) 

class ExperimentAnalyze(object):
    def __init__(self, args, dirs, Num=None):

        self.args = args 
        self.dirs = dirs 
        self.filename = args.model_name 
        if isinstance(dirs, dict):
            self.logPath = dirs['analyze'] 
            self.analyze = Data_Analyze(args, dirs['analyze'], Num=Num) 
        else:
            self.logPath = dirs 
            self.analyze = Data_Analyze(args, dirs, Num=Num) 
        
    def train_result_analyze(self):
        filename = self.filename+"_train_log.hdf5" 
        filepath = os.path.join( self.logPath, filename ) 
        with h5py.File(filepath, "r") as f: 
            train_loss = np.array(f['train_loss']) 
            valid_loss = np.array(f['valid_loss']) 
            train_nrmse = np.array(f['train_nrmse']) 
            valid_nrmse = np.array(f['valid_nrmse'])
            train_ndrmse = np.array(f['train_ndrmse']) 
            valid_ndrmse = np.array(f['valid_ndrmse'])
            train_time = np.array(f['train_time']) 
            if self.args.eval_standard == 'prob_acc':
                train_dist_accuracy = np.array(f['train_dist_accuracy'])
                valid_dist_accuracy = np.array(f['valid_dist_accuracy'])
        return len(train_loss), np.round(train_time,3) 
        
    def test_result_analyze(self):
        filename = self.filename+"_test_log.hdf5" 
        filepath = os.path.join( self.logPath, filename ) 
        with h5py.File(filepath, "r") as f: 
            x = np.array(f['x_input'])
            y = np.array(f['y_label'])
            yhat = np.array(f['y_predict']) 
            if self.args.eval_standard == 'prob_acc':
                test_dist_accuracy = np.array(f['test_dist_accuracy'])
            Rp = np.array(f['test_Rp'])
            test_nrmse = np.array(f['test_nrmse'])
            test_ndrmse = np.array(f['test_ndrmse']) 
            if self.args.forecast_type == 'probability': 
                sigma = np.array(f['sigma_predict'])
                df = np.array(f['df'])
                q1 = np.array(f['q1'])
                q9 = np.array(f['q9'])

        if self.args.forecast_type == 'probability':
            split_number = 1 
        else:
            logger.info(f"experiment_analyze.py: The forecast_type {self.args.forecast_type} is not defined!!!")
            os.system("pause")
            sys.exit(1)
        
        # Plot Values Distribution 
        self.analyze.plot_values_disttibution(Y=y[:, split_number:, :], Y_pred=yhat) 
        # Plot Error Distribution 
        self.analyze.plot_error_distribution(Y=y[:, split_number:, :], Y_pred=yhat) 
        # Plot Mispredictions Thresholds 
        self.analyze.plot_errors_threshold(Y=y[:, split_number:, :], Y_pred=yhat) 
        # Analyze all test point 
        self.analyze.linear_regression_kfold(Y=y.squeeze(-1)[:, split_number:], Y_pred=yhat.squeeze(-1)) 
        # Analyze all test acc distribution 
        if self.args.eval_standard == 'prob_acc':
            acc_70, acc_80, acc_90 = self.analyze.acc_all(Y=y.squeeze(-1)[:, split_number:], Y_pred=yhat.squeeze(-1))
        if self.args.forecast_type == 'probability': 
            # (batch_size, n_seq, 1)
            self.analyze.plot_prob_multi_series_forecasts_quanli(
                X=x, Y=y[:, split_number:, :].squeeze(-1), Y_pred=yhat.squeeze(-1), q1=q1, q9=q9)
            self.analyze.plot_prob_sigle_series_forecasts(
                X=x[self.args.test_number], Y=y[self.args.test_number][split_number:, :], 
                Y_pred=yhat[self.args.test_number], Sigma=sigma[self.args.test_number]) 
        # get metircs 
        tp, tn, fp, fn, auc_value = self.analyze.class_metrics(Y=y.squeeze(-1), Y_pred=yhat.squeeze(-1)) 
        # get quantile 
        if self.args.forecast_type == 'probability': 
            self.analyze.quantile(df)
            # Return values 
            sMAPE,NRMSE,NDRMSE,ND,RMSE,MAPE = df[2, 2:8] 
            p10,p50,p90,mp50 = df[0,1], df[2,1], df[4,1], round(np.mean(df[:,1]),3) 
            if self.args.eval_standard == 'prob_acc':
                return  sMAPE, NRMSE, NDRMSE, ND, RMSE, MAPE, p10, p50, p90, mp50, acc_70, acc_80, acc_90, test_dist_accuracy 
            else:
                return  sMAPE, NRMSE, NDRMSE, ND, RMSE, MAPE, p10, p50, p90, mp50 