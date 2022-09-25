import logging, os, sys 
import torch 
# Dataset 
from numpy.random import default_rng 
import numpy as np 
import math 
# import pathlib 
from ..utils.io import load_dat 
from ..data import DataLoaderProb_ 
from ..model import deepwp 

logger = logging.getLogger(__name__) 

# Count model parameters 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device(args):
    # Seed Every Thing 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed) 
    if args.detect_hardware:
        torch.set_num_threads(2)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        logger.info("use {} GPU(s)".format(torch.cuda.device_count()))
    else:
        logger.info("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
    return device 

def create_new_datasets(args, data_post=None, dirs=None, mode=None):
    # Load datas 
    logger.info(f"Creating New Datasets.")
    logger.info(f"Loading Training Datas.") 
    # Set seed for every test have the same random number 
    rng = default_rng(args.seed) 
    remove_points = math.ceil(args.remove_time / (args.delta_t*0.001))      
    n_points = math.ceil(args.timespan / (args.delta_t*0.001))              
    data = load_dat(filepath=args.filepath, mode=args.fileformat)
    # Multi Series Auto 
    if len(args.location) == 0:
        args.location = np.arange(data.shape[0])
    data = data[:, remove_points:]          # data with shape (N_series, total_len) 
    if (n_points >= data.shape[1]):
        train_data = data[args.location, ::args.sub_sampling].T # (seq_len, N_series)
    else:  
        stride = data.shape[1] - n_points
        randSplit = rng.choice(stride, size=1, replace=False)[0]
        if not (args.debug):                    
            train_data = data[args.location, randSplit:randSplit+n_points][:, ::args.sub_sampling].T  # (seq_len, N_series) 
        else: 
            if args.debug_id == None:
                logger.info(f"The Debug mode has been activated, you should set debug_id!")
                os.system("pause") 
                sys.exit(1)                       
            train_data = data[args.location, args.debug_id:args.debug_id+n_points][:, ::args.sub_sampling].T    # (n_points, 1)
        
    # Handel Datas 
    logger.info(f"Handling Datas.") 
    if args.forecast_type == 'probability': 
        hand_data = DataLoaderProb_(train=train_data, args=args, dirs=dirs, scaler=args.scaler)
    else:
        raise Warning('Not supported forecast_type')

    train_loader, valid_loader, test_loader = hand_data.train_val_test_split(mode=mode)
    return train_loader, valid_loader, test_loader, hand_data
    #=========================================== * Random Wave ============================ 

def create_model(args, model_type, hand_data, device):
    if model_type == 'deepwp':
        model = deepwp(
            d_lag=hand_data.d_lag, d_cov=hand_data.d_cov, d_emb=hand_data.emb_dim, 
            d_output=hand_data.dim_output, d_hidden=args.hidden_state, 
            dropout=args.dropout, 
            N=args.num_layers, dim_maxseqlen=hand_data.window, mode=args.covariate_mode, 
            args=args) 
    else:
        raise Warning('Not supported model')
    return model 