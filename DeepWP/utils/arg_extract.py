import argparse 
import os, sys  

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()
    # ================= Basic Configuration 
    # Random Args:
    parser.add_argument('--seed', type=int, default=12345, help='Seed to use for random number generator for experiment') 
    # 
    parser.add_argument('--main_dir', type=str, default="dummy", help='Experiment main directory') 
    parser.add_argument('--save', type=str2bool, default=False, help='Switch to save') 
    parser.add_argument('--plt_show', type=str2bool, default=True, help='Switch to plot figure') 
    parser.add_argument('--plt_multi', type=str2bool, default=False, help='Swich to plot multi forecast series')
    parser.add_argument('--forecast_type', type=str, default='probability', help='Forecasting method') 
    # Debug Args: 
    parser.add_argument('--debug', type=str2bool, default=False, help='Switch to debug') 
    # Detect Hard Ware Args:
    parser.add_argument('--detect_hardware', type=str2bool, default=False, help='Detect model and hardware')

    # ================= Dataset 
    parser.add_argument('--filename', type=str, default="dummy", help='Source data filename') 
    parser.add_argument('--filepath', type=str, default="dummy", help='Source data file directory') 
    parser.add_argument('--data_source', type=str, default="SYS", help="Source data come from")
    parser.add_argument('--Hs', type=float, default=None, help='Significant height')
    parser.add_argument('--fileformat', type=int, default=1, help='Source data file format') 
    parser.add_argument('--delta_t', type=int, default=100, help='Source data sampling delta time (ms)') 
    parser.add_argument('--remove_time', type=float, default=60, help="Remove non-stationary wave datas") 
    parser.add_argument('--timespan', type=float, default=4096, help="Timeseries will be used to training last time (s)")
    parser.add_argument('--location', type=list, default=[], help="A list of locations to predict one location") 
    parser.add_argument('--sub_sampling', type=int, default=None, help='Training data sampling rate') 
    parser.add_argument('--scaling_train', type=str2bool, default=True, help='Switch to scaling during training') 
    parser.add_argument('--scaler', type=int, default=1, help='Scaler method') 
    parser.add_argument('--test_size', type=float, default=0.1, help='Test ratio rate') 
    parser.add_argument('--valid_size', type=float, default=0.1, help='Valid ratio rate') 
    parser.add_argument('--n_dim', type=int, default=1, help='Number of features') 
    parser.add_argument('--n_lag', type=int, default=None, help='Input length')
    parser.add_argument('--n_seq', type=int, default=None, help='Target length') 
    parser.add_argument('--data_stride', type=int, default=1, help='Sampling rate') 
    parser.add_argument('--covariate_mode', type=int, default=1, help="Probability model covariate mode") 
    parser.add_argument('--n_covariate', type=int, default=None, help='Number of covariate types')
    # Debug Args: 
    parser.add_argument('--debug_id', type=int, default=None, help='Debug timeseires start point') 

    # ================= Dataloader 
    parser.add_argument('--batch_size', type=int, default=32, help='Train Batch size') 
    parser.add_argument('--batch_size_valid', type=int, default=64, help='Valid Batch size') 
    parser.add_argument('--batch_size_test', type=int, default=32, help='Test Batch size') 

    # ================= Model Basic Configuration 
    parser.add_argument('--model_configfile', type=str, default=None, help='Model config filename') 
    parser.add_argument('--model_configpath', type=str, default=None, help='Model config filepath') 
    parser.add_argument('--n_modeltype', type=int, default=None, help='Number of model types')
    parser.add_argument('--model_type', type=str, default=None, help='Network architecture for training') 
    parser.add_argument('--model_name', type=str, default="dummy",help='Model name') 

    # ================= Model Hyperparameters 
    # DeepAR Args:
    parser.add_argument('--num_layers', type=int, default=1, help='Stack number of layers')
    parser.add_argument('--hidden_state', type=int, default=40, help='Hiden size of the neural network') 
    # ================= Training Hyperparameters 
    # Basic Args:
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs') 
    parser.add_argument('--eval_standard', type=str, default='loss', help='Save best training model mode') 
    parser.add_argument('--early_stop', type=str2bool, default=True, help='If the loss or acc is not change, quit the training') 
    # Keep Best Args:
    parser.add_argument('--keep_best', type=str2bool, default=False, help='Keep more than 1 model in models directory')
    parser.add_argument('--keep_bestNum', type=int, default=5, help='The number of models keep')
    # Optimizer Args: 
    parser.add_argument('--optimizer_mode', type=int, default=1, help='Optimizer mode') 
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate') 
    parser.add_argument('--weightDecay', type=float, default=1e-8, help='Help to reduce overfitting')
    parser.add_argument('--scheduler_dynamic', type=str2bool, default=False, help='Dynamic learning rate')
    # ReduceLROnPlateau Args: 
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Epoch patience before reducing learning_rate')
    parser.add_argument('--scheduler_factor', type=float, default=0.8, help='Factor to reduce learning_rate') 
    # ExponentialLR Args: 
    parser.add_argument('--scheduler_gamma', type=float, default=0.995, help='Factor to reduce learning_rate') 
    # Probability Class Args:
    parser.add_argument('--pdf', type=str, default='gaussian', help='Probabaility loss function') 

    # ================= Testing Hyperparameters 
    parser.add_argument('--test_number', type=int, default=0, help='Random numbet to test programing') 
    parser.add_argument('--test_all', type=str2bool, default=True, help='Calculate all test error')

    # ================== Initialize Args 
    args = parser.parse_args()

    return args 