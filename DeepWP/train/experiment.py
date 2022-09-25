import logging, os, sys 
import torch 
from ..utils.io import save, save_json
from .experiment_tools import get_device 

logger = logging.getLogger(__name__) 

class Experiment():
    def __init__(self, args, kth_fold=None):
        logger.info('Experiment %s' % args.model_name)
        self.args = args 
        self.device = get_device(args) 
        self._filesystem_structure(kth_fold) 
        self._mkdirs() 

    def _create_scheduler(self):
        if self.args.optimizer_mode == 1:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate, 
                weight_decay=self.args.weightDecay
            )

            if self.args.eval_standard == 'loss' or self.args.eval_standard == 'ndrmse': 
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer = self.optimizer, mode='min', 
                    factor=self.args.scheduler_factor, patience=self.args.scheduler_patience)
            elif self.args.eval_standard == 'prob_acc':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer = self.optimizer, mode='max', 
                    factor=self.args.scheduler_factor, patience=self.args.scheduler_patience) 
            else:
                logger.info("You should check the code")
                os.system("pause")
                sys.exit(1)
        elif self.args.optimizer_mode == 2:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate*self.args.scheduler_gamma,
                weight_decay=self.args.weightDecay 
            )

            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self.args.scheduler_gamma 
            )
    
    def _save_metadata(self):
        meta_data_dict = {  "args": vars(self.args),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "model": "%s" % self.model
                        }
        save(meta_data_dict, self.files['metadata'])
        save_json(meta_data_dict, self.files['metadata'] + '.json')

    def _mkdirs(self):
        logger.info('Creating directories')
        # Main Directory 
        if not os.path.isdir(self.dirs['experiments']):
            os.mkdir(self.dirs['experiments']) 
        # Config Directory 
        if not os.path.isdir(self.dirs['config']): 
            os.mkdir(self.dirs['config']) 

        if not self.args.detect_hardware:
            # Sub-model Main Directory 
            if not os.path.isdir(self.dirs['model_type']):
                os.mkdir(self.dirs['model_type'])
            # Sub-model Main Directory 
            if not os.path.isdir(self.dirs['results']):
                os.mkdir(self.dirs['results'])
            # Sub-model Info Directory 
            for d in self.sub_folders:
                if not os.path.isdir(self.dirs[d]):
                    os.mkdir(self.dirs[d])

    def _filesystem_structure(self, k_foldNum=None):
        self.dirs = {}
        self.dirs['experiments'] = self.args.main_dir                                           # Main directory 
        if self.args.model_configpath is not None:
            _, config_file = os.path.split(self.args.model_configpath)
            filename, _ = os.path.splitext(config_file) 
            self.dirs['config'] = os.path.join(self.dirs['experiments'], filename) 
        else:
            self.dirs['config'] = os.path.join(self.dirs['experiments'], "single")

        if not self.args.detect_hardware:
            sub_main_dir = f"{self.args.data_source}_{self.args.n_lag}_{self.args.n_seq}_{self.args.sub_sampling}"+\
                f"_{self.args.covariate_mode}"
            self.dirs['model_type'] = os.path.join(self.dirs['config'], sub_main_dir) 
            self.dirs['results'] = os.path.join(self.dirs['model_type'], self.args.model_name)      # Sub-Model Results directory

            self.sub_folders = ['pickles', 'models', 'charts', 'analyze']                           # Sub-Model Info directory 
            for d in self.sub_folders: 
                self.dirs[d] = os.path.join(self.dirs['results'], '%s\\' % d)
            self.args.model_filepath = self.dirs['charts'] 

            self.files = {}
            self.files['metadata'] = os.path.join(self.dirs['pickles'], "metadata.pickle")