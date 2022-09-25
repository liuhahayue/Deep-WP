from .experiment import Experiment 
from .experiment_runner import ExperimentRunner 
from .experiment_analyze import ExperimentAnalyze 
import logging, os, sys   
import pandas as pd 
import numpy as np 
import datetime 
from ..data import Data_Post 
from ..data.tools import fix_seed 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from .experiment_tools import get_device, create_model, create_new_datasets, count_parameters   
from ..train.experiment_analyze import KFlodsAnalyze 

logger = logging.getLogger(__name__) 

class MultiExperiments(Experiment):
    def __init__(self, args):
        self.args = args 

    # Create A New Model Directory 
    def _create_new_directory(self, kth_fold=None):
        # Create files structure 
        self._filesystem_structure(kth_fold) 
        # Create directories 
        self._mkdirs()

    # Read experiment csv 
    def _read_table(self, filename):
        if os.path.isfile(filename):
            table = pd.read_csv(filename, sep=';') 
        else:
            raise FileNotFoundError("Provided path ({}) should be a file".format(filename))
        return table 
    
    # Initialize Config 
    def _initialize_config(self, idx): 
        # Model hyperparameters 
        self.algorithm = self.table.loc[idx, 'algorithm'] 
        if self.algorithm is None:
            logger.info("You should set model type to train") 
            os.system("pause")
            sys.exit(1)
        self.args.num_layers = int(self.table.loc[idx, 'N']) 
        self.args.hidden_state = int(self.table.loc[idx, 'd_hidden']) 
        self.args.dropout = self.table.loc[idx, 'dropout'] 
        # Training hyperparameters 
        self.args.covariate_mode = int(self.table.loc[idx, 'covariate_mode']) 
        self.args.learning_rate = self.table.loc[idx, 'learning_rate'] 
        # Dataloader 
        batch_size = int(self.table.loc[idx, 'batch_size']) 
        self.args.batch_size = batch_size 
        self.args.batch_size_valid = batch_size 
        self.args.batch_size_test = batch_size 

    # Loop Model 
    def LoopModels(self): 
        # Load Model Config Files 
        self.table = self._read_table(filename=self.args.model_configpath)
        self.args.detect_cycle = self.table.shape[0]
        self.args.detect_cycle_id = 0 
        # Total result 
        total_analyze = KFlodsAnalyze(self.args, self.table.shape[0])

        # Loop Model 
        self.args.sub_sampling, self.args.n_lag, self.args.n_seq = None, None, None 
        for i in range(self.table.shape[0]):
            self.device = get_device(self.args) 
            # Read experiment table, set hyperparameters 
            idx = self.table[self.table['in_progress'] == -1].isnull()['score'].idxmax() 
            self.args.detect_cycle_id += 1 
            # Config Initialize 
            self._initialize_config(idx)
            # Dataset 
            n_lag = self.table.loc[idx, 'n_lag']
            n_seq = self.table.loc[idx, 'n_seq']
            sub = self.table.loc[idx, 'sub'] 
            if np.isnan(n_lag) or np.isnan(n_seq) or np.isnan(sub):
                self.table.loc[idx, 'in_progress'] = self.device 
                continue 
            if (self.algorithm == self.args.model_type) and (sub == self.args.sub_sampling) \
                and (n_lag == self.args.n_lag) and (n_seq == self.args.n_seq):
                pass
            else:
                # Create New datasets 
                self.args.model_type = self.algorithm 
                self.args.sub_sampling, self.args.n_lag, self.args.n_seq = int(sub), int(n_lag), int(n_seq) 
                train_set, valid_set, test_set, self.hand_data = create_new_datasets(
                    args=self.args, data_post=None, dirs=None, mode='multi') 
            # Dataloader 
            # Fix seed 
            fix_seed(seed=self.args.seed) 
            self.training_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
            self.validing_loader = DataLoader(dataset=valid_set, batch_size=self.args.batch_size_valid, shuffle=True, drop_last=False)
            self.testing_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size_test, shuffle=False, drop_last=False)

            # Making Result Directories 
            Prob_type1 = ['deepwp']
            if self.args.model_type in Prob_type1:
                self.args.model_name = f"{self.args.model_type}_{self.args.hidden_state}_{self.args.batch_size}_{self.args.learning_rate}"
            else:
                logger.info('The model is not in current training mode')
                os.system("pause")
                sys.exit(1)
            
            if not self.args.detect_hardware:
                self._create_new_directory() 
            elif self.args.detect_hardware and i == 0:
                self._create_new_directory() 
            self.filepath = self.dirs['config'] 
            logger.info('Experiment %s' % f"{self.args.model_name}_{self.args.covariate_mode}") 
            self.table.loc[idx, 'in_progress'] = self.device 

            # Training loop 
            # Data 
            if not self.args.detect_hardware:
                self.data_post = Data_Post(args=self.args, dirs=self.dirs) 
            else:
                self.data_post = Data_Post(args=self.args, dirs=None) 
            # Detect hard ware 
            if self.args.detect_hardware:
                start_time = f"{datetime.datetime.now()}"

            # Model 
            self.model = create_model(self.args, self.args.model_type, self.hand_data, self.device)
            self._create_scheduler() 
            self.mseLoss = nn.MSELoss() 
            if not self.args.detect_hardware:
                self._save_metadata()
            self.model.to(self.device) 

            # ===================== Train and Test 
            experiment_runner = ExperimentRunner(experiment=self) 
            # Train 
            if self.args.detect_hardware:
                epoch_time = experiment_runner.run_experiment()
            else:
                experiment_runner.run_experiment()

            if self.args.detect_hardware:
                end_time = f"{datetime.datetime.now()}" 
                parameters = count_parameters(self.model)
                total_analyze.result_df.loc[idx, 'Model'] = f"{self.args.model_name}_{self.args.covariate_mode}_{self.args.n_lag}_{self.args.n_seq}"
                total_analyze.result_df.loc[idx, 'start_time'] = start_time 
                total_analyze.result_df.loc[idx, 'end_time'] = end_time 
                total_analyze.result_df.loc[idx, 'epoch_time'] = epoch_time 
                total_analyze.result_df.loc[idx, 'parameters'] = int(parameters)  
            else:
                # Test 
                experiment_runner.test_experiment()
                # ================= Analyze 
                # Analyze 
                experiment_analyze = ExperimentAnalyze(args=self.args, dirs=self.dirs) 
                # Train analyze 
                total_analyze.kflods_train_analyze(experiment_analyze)
                # Test analyze 
                total_analyze.kflods_test_analyze(experiment_analyze, idx)

            # Release memory 
            torch.cuda.empty_cache() 
            total_analyze.load_k_fold_best(total_analyze, idx, self.dirs['config'])

        logger.info('All models have been trained!!!') 
        if not self.args.detect_hardware:
            total_analyze.output_k_fold_best(self.dirs['config'])