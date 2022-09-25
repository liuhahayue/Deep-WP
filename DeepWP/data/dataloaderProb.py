"""
Dataloader 
"""
import torch 
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import Dataset 
import numpy as np 
import pandas as pd 
from .tools import fix_seed, time_features   
import os, sys, logging   

################################################################
#  dataset
################################################################
class ProbDataset(Dataset): 
    def __init__(self, input_data, lable_data, index): 
        self.input_data = input_data 
        self.lable_data = lable_data 
        self.index = index 

    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx):
        x = self.input_data[self.index[idx, 0], self.index[idx, 1]] 
        y = self.lable_data[self.index[idx, 0], self.index[idx, 1]] 

        return x, y 

class DataLoaderProb_(object): 
    """
    Parent clss for handle data 

    Parameters:
    train (np.array) : (seq_len, N_series)
    args : Config parameters 
    scaler (int) : if scaler == 1, default scaling method StandardScaler()
                   if scaler == 2, scaling method MinMaxScaler(feature_range=(0, 1)) 
    """ 
    def __init__(self, train, args, dirs, scaler=None):
        self.train = train 
        self.args = args 
        self.dirs = dirs
        self.window = args.n_lag + args.n_seq 
        self.p_train = 1 - args.valid_size - args.test_size 
        self.p_validate = args.valid_size 

        # Scaling 
        self.scaled_data = self.train 

    def train_val_test_split(self, mode=None): 
        # numpy 2 dataframe 
        df = pd.DataFrame(self.scaled_data) 
        # 时间戳 索引 
        freq = self.args.sub_sampling * self.args.delta_t     # (unit:ms) 
        periods = self.scaled_data.shape[0] 
        data_timestamp = pd.date_range(start='2022-1-11', periods=periods, freq=str(freq)+'ms')     # fake global timestamp 
        df = df.set_index(pd.to_datetime(data_timestamp))
        # Create index for allowable entries (not only zeros) 
        self.num_series = len(df.columns)       #   1
        self.num_dates = len(df)                #   12800 
        index = np.empty((0, 2), dtype='int') 
        for i in range(len(df.columns)):
            idx = np.flatnonzero(df.iloc[:, i])     # (12795, ) 
            arr = np.array([np.repeat(i, len(idx)), idx]).T # 12800 
            index = np.append(index, arr, axis = 0)         # 

        # Stack (and recreate Dataframe because stupid Pandas creates a Series otherwise) 
        df = pd.DataFrame(df.stack()) 
        # Add categorical covariates 
        df['Series'] = df.index.get_level_values(1).astype('int') 
        # Add time-based covariates 
        data_stamp = time_features(dates=df.index.get_level_values(0), timeenc=1, freq='us') 
        df['MicrosecondOfSecond'] = data_stamp[:, 0] 
        df['SecondOfMinute'] = data_stamp[:, 1] 
        df['MinuteOfHour'] = data_stamp[:, 2]
        # Rename target column 
        df.rename(columns={0:'Wave_elevation'}, inplace=True) 
        # Add lagged output variable 
        df['Wave_elevation_lag'] = df.groupby(level=1)['Wave_elevation'].shift(1)
        # Remove the last timestep (contains NaNs for the lagged column) 
        df = df.iloc[self.num_series:]
        # Sort by series 
        df.index.names = ['time','series']
        df.sort_index(level=['series','time'], inplace=True) 

        # Create feature matrix X and output vector Y 
        # Wave_elevation   
        df_Y = df[['Wave_elevation']] 
        df.drop(['Wave_elevation'], axis=1, inplace=True) 
        # Series  MicrosecondOfSecond  SecondOfMinute  MinuteOfHour Wave_elevation_lag
        df_X = df 
        # Distribution check 
        df_distribution = df['Wave_elevation_lag'].values
        # Convert dataframe to numpy and reshape to [series x timestep x features] format 
        X = df_X.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_X.columns))  # (14, 12799, 5) (series, tiemstep, features)
        Y = df_Y.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_Y.columns))  # (14, 12799, 1) (series, tiemstep, features) 
        # Input and output dimensions 
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]              # (5) (1) 
        # Convert to torch 
        X, Y = torch.from_numpy(X), torch.from_numpy(Y) 
        # Create subsequences by unfolding along date dimension with a sliding window 
        # (series, tiemstep, window, features)
        Xt, Yt = X.unfold(-2, self.window, self.args.data_stride).permute(0, 1, 3, 2), \
            Y.unfold(-2, self.window, self.args.data_stride).permute(0, 1, 3, 2) # (1, 12776, 24, 5) 
        # Create train, validate and test sets 
        num_dates_train = int(self.p_train * Xt.shape[1])                       # 10220 
        num_dates_validate = int(self.p_validate * Xt.shape[1])                 # 1277 

        # Get datasets 
        index = torch.from_numpy(index)                                         # ([**, 2(customer_id, idx)]) 
        train_index = index[index[:, 1] < num_dates_train]                      # (10216 * 14, 2)
        valid_index = index[(index[:, 1] >= num_dates_train) & (index[:, 1] < num_dates_train + num_dates_validate)] # (1277*14, 2)
        test_index = index[(index[:, 1] >= num_dates_train + num_dates_validate + self.window - 1) & (index[:, 1] < Xt.shape[1])] # (1248*17, 2)

        x_input = Xt 
        y_label = Yt[:, :, self.args.n_lag:self.window] 

        # Dataset 
        dataset_train = ProbDataset(input_data=x_input, lable_data=y_label, index=train_index)
        dataset_valid = ProbDataset(input_data=x_input, lable_data=y_label, index=valid_index) 
        dataset_test = ProbDataset(input_data=x_input, lable_data=y_label, index=test_index)

        # fix seed 
        fix_seed(seed=self.args.seed)
        # Initialize sample sets 
        num_samples_train = 500000 
        num_samples_validate = 10000 
        num_samples_test = 10000 
        id_samples_train = torch.randperm(len(dataset_train))[:num_samples_train] 
        id_samples_validate = torch.randperm(len(dataset_valid))[:num_samples_validate]
        id_samples_test = torch.randperm(len(dataset_test))[:num_samples_test]
        # Subset 
        train_data_subset = torch.utils.data.Subset(dataset_train, id_samples_train) 
        valid_data_subset = torch.utils.data.Subset(dataset_valid, id_samples_validate) 
        test_data_subset = torch.utils.data.Subset(dataset_test, id_samples_test)

        # fix seed 
        fix_seed(seed=self.args.seed)
        # Dataloader 
        if mode == 'multi':
            pass
        else:
            train_loader = DataLoader(dataset=train_data_subset, batch_size=self.args.batch_size, shuffle=True, drop_last=False) 
            valid_loader = DataLoader(dataset=valid_data_subset, batch_size=self.args.batch_size_valid, shuffle=True, drop_last=False) 
            test_loader = DataLoader(dataset=test_data_subset, batch_size=self.args.batch_size_test, shuffle=False, drop_last=False)

        # Useful for use in algorithms - dimension of lags and dimension of covariates (minus dim of time series ID) 
        self.d_lag = self.dim_output    # number of lags in input               # (1) 
        self.d_emb = 1                                                          # (1)   # cat 
        self.d_cov = self.dim_input - self.dim_output - 1                       # (3) 
        self.emb_dim = np.array([[self.num_series, 20]])                        
        if mode == 'multi':
            return train_data_subset, valid_data_subset, test_data_subset 
        else:
            return train_loader, valid_loader, test_loader 