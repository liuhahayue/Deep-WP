"""
loss functions 
"""
import numpy as np 
import pandas as pd 
import os, sys 
from scipy.stats import norm, entropy 
import bisect 

# Error 
def QuantileLoss(y, yhat, quantile):       
    # Source: https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py
    return np.divide((2.0 * np.sum(np.abs((yhat - y) * ((y <= yhat) - quantile)))) , np.sum(np.abs(y)))

def sMAPE(y, yhat):
    # Source: https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py
    denominator = np.abs(y) + np.abs(yhat)
    flag = denominator == 0
    smape = 2 * np.mean((np.abs(y - yhat) * (1 - flag)) / (denominator + flag))
    return smape

def RMSE(y, yhat):
    # Source: https://github.com/elephaint/pedpf/blob/master/lib/loss_metrics.py
    loss = np.sqrt(np.mean(np.square(y - yhat)))
    return loss 

def NRMSE(y, yhat):
    # Source: https://github.com/elephaint/pedpf/blob/master/lib/loss_metrics.py 
    loss_rmse = RMSE(y, yhat)
    yabsmean = np.mean(np.abs(y))
    flag = yabsmean == 0
    return np.divide(loss_rmse * (1 - flag), yabsmean + flag)

def ND(y, yhat):
    # Source: https://github.com/elephaint/pedpf/blob/master/lib/loss_metrics.py
    abs_error = np.sum(np.abs(y - yhat))
    yabssum = np.sum(np.abs(y))
    flag = yabssum == 0  
    return np.divide(abs_error * (1 - flag), yabssum + flag) 

def MAPE(y, yhat):
    # Source: https://github.com/elephaint/pedpf/blob/master/lib/loss_metrics.py
    yhat = yhat[y != 0]
    y = y[y != 0]
    loss = 1 / y.shape[0] * np.sum( np.abs( (yhat - y) / y ) )
    return loss

# 
def NDRMSE(y, yhat, Hs, verbose=None):
    y = y.transpose(1,0)        # (batch_size, n_seq)
    yhat = yhat.transpose(1,0)
    loss_rmse = np.zeros(shape=y.shape[0])
    for i in range(y.shape[0]):
        loss_rmse[i] = np.divide(RMSE(y[i], yhat[i]), Hs)
    if verbose:
        return loss_rmse
    else:
        return np.mean(loss_rmse)

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*entropy(p, M, base=2)+0.5*entropy(q, M, base=2) 
def check_value(x): 
    return np.mean(x, axis=1), np.std(x, axis=1), np.min(x, axis=1), np.max(x, axis=1) 
def normal_distribution(loc, scale):
    return norm(loc, scale)
def cut_distribution(distr, loc):
    n_points = 5000
    mid_points = n_points // 2 
    y_mid = np.linspace(loc-5, loc+5, n_points)
    norm_pdf = distr.pdf(y_mid)
    a = np.sort(norm_pdf[-mid_points:], axis=0) 
    y_right = bisect.bisect(a, 0.00000001) 
    if y_right > 100:
        y_right = 100
    delta_y = y_mid[1] - y_mid[0]
    loc_left = loc-(mid_points-y_right)*delta_y 
    loc_right = loc + (mid_points-y_right)*delta_y 
    return loc_left, loc_right

def show_mean_prob_accuracy(y, yhat, args, mode=None):
    y = y.transpose(1,0)            # (batch, n_seq)
    y_locs, y_scales, y_mins, y_maxs = check_value(x=y)
    yhat = yhat.transpose(1,0)
    yhat_locs, yhat_scales, yhat_mins, yhat_maxs = check_value(x=yhat)
    dist_acc = np.zeros(y_locs.shape[0]) 
    for i in range(y_locs.shape[0]):
        y_dist = normal_distribution(loc=y_locs[i], scale=y_scales[i]) 
        yhat_dist = normal_distribution(loc=yhat_locs[i], scale=yhat_scales[i]) 
            
        # Mid 两个分布需要在同一个区间采样 
        loc_left_y, loc_right_y = cut_distribution(y_dist, y_locs[i]) 
        loc_left_yhat, loc_right_yhat = cut_distribution(yhat_dist, yhat_locs[i]) 
        sort_split = sorted([loc_left_y, loc_right_y, loc_left_yhat, loc_right_yhat]) 
        x_mid = np.linspace(sort_split[0], sort_split[-1], 1000)
        # PDF 
        y_pdf = y_dist.pdf(x_mid) 
        yhat_pdf = yhat_dist.pdf(x_mid) 
        # Score 
        dist_acc[i] = np.clip(JS_divergence(y_pdf, yhat_pdf), 0, 1)
        if np.isnan(dist_acc[i]):
            dist_acc[i] = 1
    dist_acc = (1 - dist_acc)*100.0     # js 散度越小越好 
    if mode == 'analyze':
        return dist_acc
    else:
        return np.mean(dist_acc) 

def metrics(yhat, y, quantiles, mode, args):
    """
    Args:
        yhat (numpy.array) : (num_forecasts, n_seq, num_samples) 
        y (numpy.array)    : (n_seq, num_samples) 
        quantiles (numpy.array) : (num_forecasts,)
    """
    if args.eval_standard == 'prob_acc':
        columns = ['Quantile', 'QuantileLoss', 'sMAPE', 'NRMSE', 'NDRMSE', 'ND', 'RMSE','MAPE','Prob_acc'] 
    else:
        columns = ['Quantile', 'QuantileLoss', 'sMAPE', 'NRMSE', 'NDRMSE', 'ND', 'RMSE','MAPE'] 
    data = np.zeros((len(quantiles), len(columns)))         # (rows, cols)
    df = pd.DataFrame(data, columns=columns)
    for q, quantile in enumerate(quantiles):
        df.loc[q, 'Quantile'] = quantiles[q]
        df.loc[q, 'QuantileLoss'] = QuantileLoss(y, yhat[q], quantile) 
        df.loc[q, 'sMAPE'] = sMAPE(y, yhat[q]) 
        df.loc[q, 'NRMSE'] = NRMSE(y, yhat[q])
        df.loc[q, 'NDRMSE'] = NDRMSE(y, yhat[q], args.Hs)
        df.loc[q, 'ND'] = ND(y, yhat[q]) 
        df.loc[q, 'RMSE'] = RMSE(y, yhat[q])
        df.loc[q, 'MAPE'] = MAPE(y, yhat[q])
        if args.eval_standard == 'prob_acc':
            if mode == 'train':
                df.loc[q, 'Prob_acc'] = None
            else:
                df.loc[q, 'Prob_acc'] = show_mean_prob_accuracy(y, yhat[q], args)
    
    q = 2           # q = 4 是表示中位数为5 
    if args.eval_standard == 'prob_acc':
        return df, df['QuantileLoss'][q], None, df['Prob_acc'][q], df['NRMSE'][q], df['NDRMSE'][q]   
    else:
        return df, df['QuantileLoss'][q], None, None, df['NRMSE'][q], df['NDRMSE'][q] 