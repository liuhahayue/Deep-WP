import matplotlib as mpl 
import matplotlib.pyplot as plt 
import os, logging, sys   
import numpy as np 
import h5py 

logger = logging.getLogger(__name__) 

class Data_Post(object):
    """ 
    Parent class for post-processing datas 
    """
    def __init__(self, args, dirs, dpi=100, fig_extension="png", Num=None):
        if not args.detect_hardware:
            self.filepath = dirs['charts']
            self.logPath = dirs['analyze'] 
        else:
            self.filepath = None
            self.logPath = None 
        self.args = args 
        self.filename = args.model_name
        self.save = args.save 
        self.dpi = dpi 
        self.fig_extension = fig_extension 

    def save_fig(self, filename):
        if os.path.isfile(self.filepath):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.filepath))
        os.makedirs(name=self.filepath, exist_ok=True)

        path = os.path.join(self.filepath, filename + "." + self.fig_extension)
        plt.tight_layout()
        plt.savefig(path, format=self.fig_extension)
        if self.args.plt_show:
            plt.show()          # by liuhahayue 
        else:   
            plt.close()         # 将图片直接关闭不需要展示 

    def plot_learning_curves(self, loss, val_loss, epochs):
        plt.plot(np.arange(len(loss)), loss, "b.-", label="Training loss")
        plt.plot(np.arange(len(val_loss)), val_loss, "r.-", label="Validation loss")
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        max_val = np.max(np.concatenate((loss, val_loss), axis=0))*1.25
        min_val = np.min(np.concatenate((loss, val_loss), axis=0))
        if min_val > 0:
            min_val *= 0.75 
        else:
            min_val *= 1.25 
        if self.args.forecast_type == 'probability':
            plt.axis([-0.25, epochs-0.75, min_val, max_val])
        else:
            plt.axis([-0.25, epochs-0.75, 0, max_val])

        plt.legend(loc='upper right', fontsize=12)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        name = self.filename+"_loss"
        plt.title(label = name)
        if self.save:
            self.save_fig(filename=name)

    def plot_accuracy_curves(self, acc, val_acc, epochs, mode):
        if mode == 'prob': 
            max_val = np.max(val_acc)*1.25 
            acc = np.array([np.mean(val_acc)]*len(val_acc))
            plt.plot(np.arange(len(acc)), acc, "b.-", label="Prob mean acc") 
        else:
            max_val = np.max(np.concatenate((acc, val_acc), axis=0))*1.25
            plt.plot(np.arange(len(acc)), acc, "b.-", label="Training acc")
        plt.plot(np.arange(len(val_acc)), val_acc, "r.-", label="Validation acc")
        if mode == 'prob':
            plt.plot(epochs-1, self.test_dist_acc, "k*", markersize=10, markerfacecolor='none', label="Test acc") 
        else:
            plt.plot(epochs-1, self.test_acc, "k*", markersize=10, markerfacecolor='none', label="Test acc")
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        plt.axis([-0.25, epochs-0.75, 0, max_val])
        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel("Epochs")
        plt.ylabel("Acc")
        plt.grid(True)
        name = self.filename+f"_accuracy_{mode}"
        plt.title(label = name)
        if self.save:
            self.save_fig(filename=name)

    def plot_nrmse_curves(self, nrmse, val_nrmse, epochs, mode=None):
        max_val = np.max(np.concatenate((nrmse, val_nrmse), axis=0))*1.25
        plt.plot(np.arange(len(nrmse)), nrmse, "b.-", label=f"Training {mode}")
        plt.plot(np.arange(len(val_nrmse)), val_nrmse, "r.-", label=f"Validation {mode}")
        if mode == 'ndrmse':
            plt.plot(epochs-1, self.test_ndrmse, "k*", markersize=10, markerfacecolor='none', label=f"Test {mode}")
        else:
            logger.info("You should check the code")
            os.system("pause")
            sys.exit(1)
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        plt.axis([-0.25, epochs-0.75, 0, max_val])
        plt.legend(loc='upper right', fontsize=12)
        plt.xlabel("Epochs")
        if mode == 'ndrmse':
            plt.ylabel("NDRMSE")
        plt.grid(True)
        name = f"{self.filename}_{mode}"
        plt.title(label = name)
        if self.save:
            self.save_fig(filename=name)

    def testing_log(self, x, y, yhat, acc, dist_acc, 
        Rp, nrmse, ndrmse, sigma=None, df=None, q1=None, q9=None):
        if self.args.eval_standard == 'prob_acc':
            # Plot Dist Accuracy Curve 
            self.test_dist_acc = dist_acc 
            self.plot_accuracy_curves(acc=self.dist_acc, val_acc=self.val_dist_acc, epochs=self.epochs, mode='prob') 
        else:
            self.test_ndrmse = ndrmse 
            self.plot_nrmse_curves(nrmse=self.ndrmse, val_nrmse=self.val_ndrmse, epochs=self.epochs, mode='ndrmse')

        # Output train log 
        if os.path.isfile(self.logPath):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.logPath))
        os.makedirs(name=self.logPath, exist_ok=True) 
        filename = self.filename+"_test_log.hdf5"
        filepath = os.path.join( self.logPath, filename ) 

        if os.path.exists(filepath):
            print(f"The file is exsisting in {self.logPath}, the old file will be overwritten!") 

        with h5py.File(filepath, "w") as f: 
            f['x_input'] = x 
            f['y_label'] = y  
            f['y_predict'] = yhat 
            if self.args.eval_standard == 'prob_acc':
                f['test_dist_accuracy'] = dist_acc 
            f['test_Rp'] = Rp  
            f['test_nrmse'] = nrmse 
            f['test_ndrmse'] = ndrmse  
            if self.args.forecast_type == 'probability': 
                f['sigma_predict'] = sigma 
                f['df'] = df 
                f['q1'] = q1 
                f['q9'] = q9 

    def training_log(self, loss, val_loss, rp, val_rp, 
        acc, val_acc, dist_acc, val_dist_acc, 
        nrmse, val_nrmse, ndrmse, val_ndrmse, epochs, time):
        
        # Plot Learning Loss Curve 
        self.plot_learning_curves(loss, val_loss, epochs) 
        if self.args.eval_standard == 'prob_acc': 
            self.dist_acc = dist_acc
            self.val_dist_acc = val_dist_acc 
        self.rp = rp 
        self.val_rp = val_rp 
        self.nrmse = nrmse 
        self.val_nrmse = val_nrmse 
        self.ndrmse = ndrmse 
        self.val_ndrmse = val_ndrmse 
        self.epochs = epochs 

        # Output train log 
        if os.path.isfile(self.logPath):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.logPath))
        os.makedirs(name=self.logPath, exist_ok=True)
        filename = self.filename+"_train_log.hdf5"
        filepath = os.path.join( self.logPath, filename )

        if os.path.exists(filepath):
            print(f"The file is exsisting in {self.logPath}, the old file will be overwritten!")

        with h5py.File(filepath, "w") as f: 
            f['train_loss'] = loss 
            f['valid_loss'] = val_loss 
            f['train_nrmse'] = nrmse  
            f['valid_nrmse'] = val_nrmse  
            f['train_ndrmse'] = ndrmse   
            f['valid_ndrmse'] = val_ndrmse   
            f['train_time'] = time 
            if self.args.eval_standard == 'prob_acc':
                f['train_dist_accuracy'] = dist_acc 
                f['valid_dist_accuracy'] = val_dist_acc 