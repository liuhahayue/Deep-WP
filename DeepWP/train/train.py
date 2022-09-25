import os, logging, sys  
from abc import abstractmethod 
import warnings 
from .train_tools import Validation 
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__) 

class trainer(object):
    """
    Parent class for training 
    """

    def __init__(self, model, device, args, dirs, hand_data):
        self.model = model.to(device) 
        self.device = device 
        self.args = args
        self.filename = args.model_name 
        if not args.detect_hardware:
            self.filepath = dirs['models'] 
        else:
            self.filepath = None 
        self.hand_data = hand_data 
        if self.args.keep_best:
            self.valid_keep = Validation(
                filepath=self.filepath, filename=self.filename, keep_best=self.args.keep_bestNum)

    @abstractmethod
    def train(self, epochs, optimizer, scheduler, criterion, training_loader, validing_loader):
        raise NotImplementedError("Train function has not been properly overridden") 

    @abstractmethod 
    def test(self, testing_loader, criterion):
        raise NotImplementedError("Test function has not been properly overridden") 

    def _save_best_models(self, metrics, optimizer, epoch):
        save_min = ['loss', 'ndrmse']
        save_max = ['prob_acc']
        save_flag = False 
        if self.args.eval_standard in save_min:
            if metrics < self.valid_min:
                self.valid_min = metrics 
                save_flag = True 
                if not self.args.keep_best:
                    self.model.save_model(name=self.filename, save_directory=self.filepath)
            if self.args.keep_best:
                self.valid_keep.save_best_min(metrics=metrics, env=self, epoch=epoch+1)
        elif self.args.eval_standard in save_max:
            if metrics > self.valid_max: 
                self.valid_max = metrics 
                save_flag = True 
                if not self.args.keep_best: 
                    self.model.save_model(name=self.filename, save_directory=self.filepath)
            if self.args.keep_best:
                self.valid_keep.save_best(metrics=metrics, env=self, epoch=epoch+1) 
        else:
            logger.info(f"The self.args.eval_standard {self.args.eval_standard} is not defined !") 
            os.system("pause")
            sys.exit(1)

        return save_flag 

    def _early_stop2(self, stop_flag):
        if stop_flag:
            if self.args.eval_standard == 'loss':
                logger.info(f'The best valid loss is {self.valid_min:.5f}')
            elif self.args.eval_standard == 'ndrmse':
                logger.info(f"The best ndrmse is {self.valid_min:.5f}") 
            elif self.args.eval_standard == 'prob_acc':
                logger.info(f"The best probatility accuracy is {self.valid_max:.5f}") 
            else:
                logger.info("You should check the code Line 109") 
                os.system("pause")
                sys.exit(1)
            self.stop_round = 0 
        else:
            self.stop_round += 1 