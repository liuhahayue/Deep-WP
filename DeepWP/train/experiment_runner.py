import torch.nn as nn 
import logging, os, sys  
from .train_probability import trainerProb 

logger = logging.getLogger(__name__) 

class ExperimentRunner(nn.Module): 
    def __init__(self, experiment, Num=None):
        super(ExperimentRunner, self).__init__() 

        self.exp = experiment 
        self.args = experiment.args 
        self.model = experiment.model 
        self.data_post = experiment.data_post 
        self.hand_data = experiment.hand_data 
        if not experiment.args.detect_hardware:
            self.resultPath = experiment.dirs['analyze'] 
        else:
            self.resultPath = None 
        self.Num = Num 

        self.train_data = experiment.training_loader    
        self.val_data = experiment.validing_loader 
        self.test_data = experiment.testing_loader 

        if self.args.forecast_type == 'probability': 
            self.training = trainerProb(
                model=experiment.model, device=experiment.device, args=experiment.args, 
                dirs=experiment.dirs, hand_data=experiment.hand_data)
        else:
            raise Warning('Not supported forecast type') 

    def run_experiment(self):
        # Train 
        train_loss, valid_loss, train_rp, valid_rp, train_acc, valid_acc, train_dist_acc, valid_dist_acc, \
            train_nrmse, valid_nrmse, train_ndrmse, valid_ndrmse,\
            epochs, train_time, epoch_time = self.training.train(
            epochs=self.args.epochs, optimizer=self.exp.optimizer, scheduler=self.exp.scheduler,
            criterion=self.exp.mseLoss, training_loader=self.train_data, validing_loader=self.val_data)

        if self.args.detect_hardware:
            return epoch_time 
        
        self.data_post.training_log(
            loss=train_loss, val_loss=valid_loss, rp=train_rp, val_rp=valid_rp,  
            acc=train_acc, val_acc=valid_acc, dist_acc=train_dist_acc, val_dist_acc=valid_dist_acc,
            nrmse=train_nrmse, val_nrmse=valid_nrmse, ndrmse=train_ndrmse, val_ndrmse= valid_ndrmse,
            epochs=epochs, time=train_time) 
                
    def test_experiment(self):
        if self.args.forecast_type == 'probability': 
            sc_test_x, sc_test_y, sc_pred_y, sc_pred_y_1, sc_pred_y_9, \
                test_acc, test_dist_acc, df, sigma, Rp05, nrmse, ndrmse = self.training.test(
                testing_loader = self.test_data) 
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not in the _test_epoch!!!")
            os.system("pause")
            sys.exit(1)
        
        # Test Result Output 
        if self.args.forecast_type == 'probability':
            self.data_post.testing_log(
                x=sc_test_x, y=sc_test_y, yhat=sc_pred_y, acc=test_acc, dist_acc=test_dist_acc, 
                Rp=Rp05, nrmse=nrmse, ndrmse=ndrmse, sigma=sigma, df=df, q1=sc_pred_y_1, q9=sc_pred_y_9) 