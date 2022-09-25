import numpy as np 
from numpy.random import default_rng 
from scipy import stats 
import matplotlib.pyplot as plt 
import os, logging, sys 
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc 
from sklearn.metrics import confusion_matrix 
from pandas.plotting import table 
from prettytable import PrettyTable 
from PIL import Image, ImageDraw, ImageFont 
import seaborn as sns 
from ..model.metrics import sMAPE, NRMSE, RMSE, MAPE, show_mean_prob_accuracy, NDRMSE 

logger = logging.getLogger(__name__) 

class Data_Analyze(object):
    """ 
    Parent class for analyzing datas 
    """
    def __init__(self, args, dirs, dpi=100, fig_extension="png", Num=None):
        self.args = args 
        self.filename = args.model_name 
        self.save = args.save 
        self.filepath = dirs
        self.dpi = dpi 
        self.fig_extension = fig_extension 
        self.t0 = args.n_lag 
        self.tp = args.n_seq 

    def save_fig(self, filename):
        if os.path.isfile(self.filepath):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.filepath))
        os.makedirs(name=self.filepath, exist_ok=True)

        path = os.path.join(self.filepath, filename + "." + self.fig_extension)
        if (len(path)) >= 250:
            split_name = len(self.args.model_name)
            new_filename = "sub" + filename[split_name:] 
            path = os.path.join(self.filepath, new_filename + "." + self.fig_extension)
        plt.tight_layout()  # 调整图片的布局 
        plt.savefig(path, format=self.fig_extension)
        if self.args.plt_show:
            plt.show()          # by liuhahayue 
        else:
            plt.close()         # 将图片直接关闭不需要展示 

    def plot_prob_sigle_series_forecasts_quanli(self, X, Y, Y_pred, q1, q9, index=None):
        t0 = self.t0 + 1 
        plt.plot(np.arange(0, t0), X[:t0], "go--", label="T0") 
        plt.plot(np.arange(t0, t0+self.tp), Y[:self.tp], "ro-", label="Actual") 
        plt.plot(np.arange(t0, t0+self.tp), Y_pred[:self.tp], "bx-", label="Forecast", markersize=10) 
        thread_up = q1[:self.tp].ravel()
        thread_low = q9[:self.tp].ravel() 
        plt.fill_between(
            np.arange(t0, t0+self.tp).ravel(), thread_up[:self.tp], thread_low[:self.tp], color='blue', alpha=0.2)
        max_val = np.max(np.concatenate((X[:t0], Y[:self.tp], Y_pred[:self.tp],q1[:self.tp],q9[:self.tp]), axis=0)) + 0.5
        min_val = np.min(np.concatenate((X[:t0], Y[:self.tp], Y_pred[:self.tp],q1[:self.tp],q9[:self.tp]), axis=0)) - 0.5 
        if self.args.eval_standard == 'prob_acc':
            Acc = round(show_mean_prob_accuracy(Y[:self.tp], Y_pred[:self.tp], self.args),3)
            plt.text(x=t0+0.5, y=(max_val-0.5), s=f"ACC: {Acc} %", fontdict={'size':'14', 'color':'k'}) 
        else:
            ndrmse = round(NDRMSE(Y[:self.tp], Y_pred[:self.tp], self.args.Hs), 3)
            plt.text(x=t0+0.5, y=(max_val-0.5), s=f"NDRMSE: {ndrmse}", fontdict={'size':'14', 'color':'k'}) 

        plt.axvline(t0, color='g', linestyle='dashed')
        plt.xlabel("Point")
        plt.ylabel("Height (m)")
        plt.legend(loc='upper right', fontsize=8) 
        plt.axis([-0.25, t0 + self.tp -0.75, min_val, max_val])
        if index:
            name = f"{self.filename}_prob_sigle_series_forecast_quanlit_{index}"
        else:
            name = f"{self.filename}_prob_sigle_series_forecast_quanlit"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, ) 

    def plot_error_metrics_hist(self, metric):
        max_value = np.max(metric)
        p001 = max_value * 1000 % 10      # 0.001 分位 
        p01 = max_value * 100 % 10 // 1   # 0.01 分位 
        p1 = max_value * 10 // 1          # 0.1 分位 
        if p001 >=5:
            if p01 < 9:
                max_val = round((p1*100 + (p01+1)*10)*0.001, 3) 
            else:
                max_val = round((p1+1)*0.1, 3)
        else:
            max_val = round((p1*100 + p01*10 + 5)*0.001, 3) 
        bins = np.round(np.linspace(0, max_val, 11),3)
        count, bins, patches = plt.hist(metric, bins=bins, alpha=0.5,)
        plt.legend(labels=[f"Total:{int(metric.shape[0]):d}"], loc='upper right', fontsize=10)
        plt.ylabel("Count")
        plt.xlabel("Metric") 
        plt.xticks(bins)
        # 直方图添加文字注释
        for p in patches:
            if p.get_height()>0:
                plt.annotate(
                    # 文字内容 
                    text=f"{p.get_height():1.0f}",
                    # 文字的位置 
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()), xycoords='data',
                    ha='center', va='center', fontsize=10, color='black', 
                    # 文字的偏移量 
                    xytext=(0,7), textcoords='offset points', clip_on=True,)
        name = f"{self.filename}_prob_error_metric_hist"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, ) 

        return count

    def plot_prob_multi_series_forecasts_quanli(self, X, Y, Y_pred, q1, q9):
        ndrmses = NDRMSE(Y.transpose(1,0), Y_pred.transpose(1,0), self.args.Hs, verbose=1)
        # Plot error_metrics distribution 
        count = self.plot_error_metrics_hist(ndrmses)
        if self.args.plt_multi: 
            # Random 
            # Set seed for every test have the same random number 
            rng = default_rng(self.args.seed) 
            randindex = rng.choice(X.shape[0], size=20, replace=False).reshape(4,5)
            self._plot_multi_series(X, Y, Y_pred, randindex, 'random', q1, q9) 
            # Class 
            index = np.argsort(ndrmses) 
            class_randindex = []
            for i in range(4):
                if i == 0:
                    start, end = 0, int(count[i]) 
                else:
                    start = end 
                    end += int(count[i]) 
                # Set seed for every test have the same random number 
                rng = default_rng(self.args.seed) 
                if (len(index[start:end])<5):
                    break 
                randindex = rng.choice(index[start:end], size=5, replace=False)
                class_randindex.append(randindex.tolist())
            if i == 0:
                pass
            else:
                self._plot_multi_series(X, Y, Y_pred, np.array(class_randindex), 'class', q1, q9) 
        else:
            self.plot_prob_sigle_series_forecasts_quanli(
                X[self.args.test_number].reshape(-1,1), Y[self.args.test_number].reshape(-1,1), 
                Y_pred[self.args.test_number].reshape(-1,1), 
                q1[self.args.test_number].reshape(-1,1),q9[self.args.test_number].reshape(-1,1)
            )

    def plot_prob_sigle_series_forecasts(self, X, Y, Y_pred, Sigma):
        self.t0 = self.t0 + 1
        plt.plot(np.arange(0, self.t0), X[:self.t0], "go--", label="T0") 
        plt.plot(np.arange(self.t0, self.t0+self.tp), Y[:self.tp], "ro-", label="Actual")
        plt.plot(np.arange(self.t0, self.t0+self.tp), Y_pred[:self.tp], "bx-", label="Forecast", markersize=10)
        y_low = (Y_pred[:self.tp]-2*Sigma[:self.tp]).ravel()
        y_up = (Y_pred[:self.tp]+2*Sigma[:self.tp]).ravel()
        plt.fill_between(
            np.arange(self.t0, self.t0+self.tp).ravel(), y_low[:self.tp], y_up[:self.tp], color='blue', 
            alpha=0.2
        )
        max_val = np.max(np.concatenate((X[:self.t0], Y[:self.tp], Y_pred[:self.tp]), axis=0)) + 0.5
        min_val = np.min(np.concatenate((X[:self.t0], Y[:self.tp], Y_pred[:self.tp]), axis=0)) - 0.5 
        if self.args.eval_standard == 'prob_acc':
            Acc = round(show_mean_prob_accuracy(Y[:self.tp], Y_pred[:self.tp], self.args),3)
            plt.text(x=self.t0+0.5, y=(max_val-0.5), s=f"ACC: {Acc} %", fontdict={'size':'14', 'color':'k'}) 
        else:
            ndrmse = round(NDRMSE(Y[:self.tp], Y_pred[:self.tp], self.args.Hs), 3) 
            plt.text(x=self.t0+0.5, y=(max_val-0.5), s=f"NDRMSE: {ndrmse}", fontdict={'size':'14', 'color':'k'}) 

        plt.axvline(self.t0, color='g', linestyle='dashed')
        plt.xlabel("Point")
        plt.ylabel("Height (m)")
        plt.legend(loc='upper right', fontsize=8) 
        plt.axis([-0.25, self.t0 + self.tp -0.75, min_val, max_val])
        name = self.filename+"_prob_sigle_series_forecast"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, )

    def _plot_errorbar(self, y, percentile=85, **kwargs):
        mean = y.mean(axis=0) 
        yerr = np.stack([
            mean - np.percentile(y, 100-percentile, axis=0),
            np.percentile(y, percentile, axis=0) - mean])
        plt.errorbar(np.arange(mean.shape[0]), mean, yerr=yerr, **kwargs)

    def plot_values_disttibution(self, Y, Y_pred, alpha=0.4, unit='', **kwargs):
        """
        Args:
            Y : np.array (batch_size, seq_len, n_dim)
            Y_pred : np.array (batch_size, seq_len, n_dim)
        """
        if self.args.forecast_type == 'probability':
            y_true, y_pred = Y[:, :self.tp].reshape(-1, self.tp), Y_pred[:, :self.tp].reshape(-1, self.tp)  # (batch_size, seq_len) 
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not defined!!!")
            os.system("pause")
            sys.exit(1)
            
        self._plot_errorbar(y=y_true, color='darkgreen', ls='dotted',
            marker='s', markeredgecolor='darkgreen', markerfacecolor='white', 
            markeredgewidth=2, linewidth=2, label='true mean')

        self._plot_errorbar(y=y_pred, color='black', ls='dotted',
            marker='s', markeredgecolor='black', markerfacecolor='white', 
            markeredgewidth=2, linewidth=2, label='prediction mean')

        sns.stripplot(data=y_true, linewidth=0, color='g', alpha=alpha, zorder=1, marker='.', s=8,)
        sns.stripplot(data=y_pred, linewidth=0, color='b', alpha=alpha, zorder=1, marker='.', s=8,)
        plt.xlabel("Point")
        plt.ylabel("Height (m)")
        plt.legend(loc='upper right', fontsize=8) 
        name = self.filename+"_values_disttibution"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, ) 
            
    def plot_error_distribution(self, Y, Y_pred, alpha=0.4, unit='', **kwargs):
        """
        Args:
            Y : np.array (batch_size, seq_len, n_dim)
            Y_pred : np.array (batch_size, seq_len, n_dim)
        """
        if self.args.forecast_type == 'probability':
            y_true, y_pred = Y[:, :self.tp].reshape(-1, self.tp), Y_pred[:, :self.tp].reshape(-1, self.tp)  # (batch_size, seq_len) 
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not defined!!!")
            os.system("pause")
            sys.exit(1)
        diff = y_true-y_pred 
        self._plot_errorbar(
            y=diff, color='black', ls='dotted',
            marker='s', markeredgecolor='black', markerfacecolor='white', 
            markeredgewidth=2, linewidth=2, label='mean'
        )
        sns.stripplot(
            data=diff, linewidth=0, color='orange',
            alpha=alpha, zorder=1, marker='.', s=8, 
        )
        plt.xlabel("Point")
        plt.ylabel("Height (m)") 
        plt.legend(loc='upper right', fontsize=8) 
        name = self.filename+"_errors_distribution"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, ) 

    def plot_errors_threshold(self, Y, Y_pred, error_band=0.1, unit='', **kwargs):
        """
        Args:
            Y : np.array (batch_size, seq_len, n_dim)
            Y_pred : np.array (batch_size, seq_len, n_dim)
        """
        if self.args.forecast_type == 'probability':
            y_true, y_pred = Y[:, :self.tp].reshape(-1, self.tp), Y_pred[:, :self.tp].reshape(-1, self.tp)  # (batch_size, seq_len) 
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not defined!!!") 
            os.system("pause")
            sys.exit(1)
        diff = y_true-y_pred 
        threshold_initial = max(
            np.sort(np.abs(diff), axis=0)[int(y_true.shape[0]*(1 - error_band))]) 
        threshold_range = np.linspace(
            threshold_initial*0.75, threshold_initial*1.25, 5) 

        for threshold in threshold_range: 
            number_mispredictions = np.where(
                np.abs(diff) > threshold, 1, 0).sum(axis=0) / y_true.shape[0] * 100 

            if threshold == threshold_initial: 
                plt.plot(number_mispredictions, '-',
                        label=f'threshold {threshold:.1f}{unit}', linewidth=5)
            else:
                plt.plot(number_mispredictions, '-',
                        label=f'threshold {threshold:.1f}{unit}')

        plt.axhline(y=error_band*100, color='black')
        plt.text(25, error_band*100+0.5, f'{int(error_band*100)}%') 
        plt.legend(loc='upper right', fontsize=8) 
        plt.xlabel("Point")
        plt.ylabel("Error (%)")
        name = self.filename+"_errors_threshold"
        plt.title(label = name) 
        if self.save:
            self.save_fig(filename=name, ) 

    def linear_regression_kfold(self, Y, Y_pred, n=24):
        """
        Args:
            Y : np.array (batch_size, n_seq)
            Y_pred : np.array (batch_size, n_seq)
        """
        if self.args.test_all:
            random_numbers = np.arange(Y.shape[0])
        else:
            # Set seed for every test have the same random number 
            rng = default_rng(self.args.seed)
            random_numbers = rng.choice(Y.shape[0], size=1, replace=False)

        if self.args.forecast_type == 'probability': 
            true = Y[random_numbers].reshape(-1,) 
            pred = Y_pred[random_numbers].reshape(-1,) 
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not defined !!!")
            os.system("pause")
            sys.exit(1)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            true, pred 
        )

        ax = plt.gca() 
        max_val = np.max(np.concatenate((true, pred), axis=0))
        min_val = np.min(np.concatenate((true, pred), axis=0))
        x = np.linspace(min_val, max_val, 100)

        # Model linear regression 
        m, c, r = slope, intercept, r_value 
        r_2 = r**2 
        ax.plot(
            x, m*x + c, linestyle='--', linewidth=1, alpha=1,
            label='$%0.3fx %0.3f$, $r^2=%0.3f$' % 
            (m, c, r_2)
        )

        # Scatter 
        ax.scatter(true, pred, 10, alpha=0.5)

        # Base linear regression 
        m_b, c_b, r_b = 1, 0, 1 
        rb_2 = r_b**2 
        ax.plot(
            x, m_b*x + c_b, color='red', linestyle='-', linewidth=1, alpha=1,
            label='Base, $%0.3fx %0.3f$, $r^2=%0.3f$' % 
            (m_b, c_b, rb_2)
        )

        # Config 
        ax.set_aspect('equal', 'box')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.minorticks_on()
        ax.set_title(self.filename+' True vs Predict Value, Base $r^2=%0.3f$' % (m_b), fontsize=12) 
        ax.set_ylabel('Predict Value', fontsize=10) 
        ax.set_xlabel('True Value', fontsize=10) 
        plt.legend(loc='upper right', fontsize=8)

        name = self.filename + "_True_vs_Predict" 
        if self.save:
            self.save_fig(filename=name)

    # Acc Distribution 
    def acc_all(self, Y, Y_pred):
        """
        Args:
            Y : np.array (batch_size, n_seq)
            Y_pred : np.array (batch_size, n_seq)
        """
        if self.args.test_all:
            random_numbers = np.arange(Y.shape[0])
        else:
            # Set seed for every test have the same random number 
            rng = default_rng(self.args.seed)
            random_numbers = rng.choice(Y.shape[0], size=1, replace=False)
        if self.args.forecast_type == 'probability':
            true = Y[random_numbers]
            pred = Y_pred[random_numbers]
        else:
            logger.info(f"The forecast_type {self.args.forecast_type} is not definded !!!")
            os.system("pause")
            sys.eixt(1)

        if self.args.eval_standard == 'prob_acc':
            acc = show_mean_prob_accuracy(true.transpose(1,0), pred.transpose(1,0), self.args, mode='analyze')
        columns = ['Model'] 
        metrics = [[self.filename]] 
        for i in range(10):
            if i == 0:
                numerator = acc[ (acc>=(i*10)) & (acc<=(i+1)*10) ]
            else:
                numerator = acc[ (acc>(i*10)) & (acc<=(i+1)*10) ]

            if self.args.eval_standard == 'prob_acc':
                columns.append('ProbAcc_'+str((i+1)*10)) 
            metrics[0].append(round((len(numerator) / (acc.shape[0])*100), 3))

        tab = PrettyTable()                                                             # 设置表头 
        tab.field_names = columns 
        for i in range(len(metrics)):
            tab.add_row(metrics[i])                                                     # 表格内容插入 
        tab_info = str(tab)
        print(tab_info)
        self.image_(tab_info=tab_info, imagename='Test Acc') 
        return metrics[0][-3:]  # 将准确率为 70-80 80-90 90-100 的占比返回 

    def classification_pred(self, y):
        """
        Args:
            y : np.array (n_seq, )
        Return:
            preds : np.array (n_seq-1, )
        """
        preds = []
        for i in range(1, len(y)):
            last_y = y[i - 1]
            curr_y = y[i]
            preds.append(curr_y - last_y > 0.0 )
        return np.array(preds)

    def plot_heatmap(self, confusion_matirx):
        plt.figure(figsize=(7, 7))
        sns.heatmap(data=confusion_matirx, linewidths=.1, annot=True, fmt='d', cmap='YlGnBu')
        plt.xlabel('Pred')
        plt.ylabel('True')
        name = self.filename+"_classification"
        plt.title(label = name)
        if self.save:
            self.save_fig(filename=name, )
    
    def save_image(self, img, filename):
        if os.path.isfile(self.filepath):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.filepath))
        os.makedirs(name=self.filepath, exist_ok=True)

        path = os.path.join(self.filepath, filename + "." + self.fig_extension)
        if (len(path)) >= 250:
            split_name = len(self.args.model_name)
            new_filename = "sub" + filename[split_name:] 
            path = os.path.join(self.filepath, new_filename + "." + self.fig_extension)
        img.save(path)

    def image_(self, tab_info, imagename):
        space = 5 
        # Image模块创建一个图片对象
        im = Image.new(mode='RGB', size=(7, 7)) 
        # ImageDraw向图片中进行操作，写入文字或者插入线条都可以
        draw = ImageDraw.Draw(im, "RGB")
        # 根据插入图片中的文字内容和字体信息，来确定图片的最终大小
        img_size = draw.multiline_textsize(text=tab_info)
        # 图片初始化的大小为10-10，现在根据图片内容要重新设置图片的大小
        im_new = im.resize(size=(img_size[0]+space*2, img_size[1]+space*2))
        # 删除没有必要的变量
        del draw
        del im 
        draw = ImageDraw.Draw(im_new, 'RGB')
        # 批量写入到图片中，这里的multiline_text会自动识别换行符
        draw.multiline_text((space,space), tab_info, fill=(255,255,255))
        if self.save:
            name = self.filename+"_"+imagename
            self.save_image(img=im_new, filename=name)
        # im_new.show()
        del draw 

    # Metric functions 
    def class_metrics(self, Y, Y_pred):
        """
        Args:
            Y : np.array (batch_size, n_seq)
            Y_pred : np.array (batch_size, n_seq)
        """

        if self.args.test_all:
            random_numbers = np.arange(Y.shape[0])
        else:
            # Set seed for every test have the same random number 
            rng = default_rng(self.args.seed)
            random_numbers = rng.choice(Y.shape[0], size=1, replace=False) 

        if self.args.forecast_type == 'probability':
            true = Y[random_numbers] 
            pred = Y_pred[random_numbers] 
        else:
            logger.info(f"The forecast_tpye {self.args.forecast_type} is not defined !!!")
            os.system("pause")
            sys.exit(1)

        # 在 pred 数组首地址插入前一个时间点的真实值 
        pred_add = []
        for i in range(true.shape[0]):
            pred_add.append(np.append(true[i, 0], pred[i, :]))
        pred_add = np.array(pred_add)

        # 循环遍历每一个测试序列 
        y_classification = []
        y_pred_classification = []
        for i in random_numbers:
            y_classification.append(self.classification_pred(true[i]))
            y_pred_classification.append(self.classification_pred(pred_add[i]))
        y_classification = np.array(y_classification).ravel() 
        y_pred_classification = np.array(y_pred_classification).ravel()

        # y_true : 真实的样本标签 | y_score : 对每个样本的预测结果 | pos_label : 正样本标签  
        # fpr : False positive rate | tpr : True positive rate | thresholds :
        fpr, tpr, thresholds = roc_curve(y_classification, y_pred_classification)
        auc_value = auc(fpr, tpr) 

        # labels = [] 可加可不加， 不加情况下会自动识别，自己定义
        C2 = confusion_matrix(y_classification, y_pred_classification)
        self.plot_heatmap(C2)

        # tn : 预测结果正确， 结果为负例 | fp : 预测结果错误， 预测结果为正例，label为负例
        # fn : 预测结果错误， 预测结果为负例，label为正例 | tp : 预测结果正确， 结果为正例
        if len(C2.ravel()) < 4:
            tp, tn, fp, fn, auc_value = 1, 1, 1, 1, 1.0 
        else: 
            tn, fp, fn, tp = C2.ravel()

            # Parameters 
            # y 的第一个元素为待预测时序的前一个时间点因此需要排除 
            mae = round(mean_absolute_error(y_true=true[:, 1:], y_pred=pred), 3)                  # Regression
            rmse = round(mean_squared_error(y_true=true[:, 1:], y_pred=pred, squared=True), 3)    # Regression 
            acc = round((tp + tn) / (tp + tn + fp + fn) * 100, 3)                          # Accuracy 
            pre = round(tp / (tp + fp), 3)                                                 # precision 
            recall = round(tp / (tp + fn), 3)                                              # recall 
            F1 = round((2 * pre * recall) / (pre + recall), 3)                             # F1-Score 
            auc_value = round(auc_value, 3)                                                # AUC 
            sen = round(recall, 3)                                                         # sensitivity == recall 
            spe = round(tn / (tn + fp), 3)                                                 # specificity 
            
            false_positive_rate = round(fp/(fp+tn+0.01), 3)
            positive_predictive_value = round(tp/(tp+fp+0.01), 3)
            negative_predictive_value = round(tn/(fn+tn+0.01), 3)

            metrics = [[self.filename, mae, rmse, acc, pre, F1, auc_value, sen, spe, ]]
            columns = ['Model','MAE', 'RMSE', 'ACC (%)', 'PRE', 'F1-Score', 'AUC', 'SEN (Recall)', 'SPE']
            
            tab = PrettyTable()                                                             # 设置表头 
            tab.field_names = columns 
            for i in range(len(metrics)):
                tab.add_row(metrics[i])                                                     # 表格内容插入 
            tab_info = str(tab)
            print(tab_info)
            self.image_(tab_info=tab_info, imagename='class_analyze')

        return tp, tn, fp, fn, auc_value
    
    def quantile(self, df): 
        # 全部结果 
        if self.args.eval_standard == 'prob_acc':
            columns = ['Model','Quantile', 'QuantileLoss', 'sMAPE', 'NRMSE', 'NDRMSE','ND', 'RMSE','MAPE','Prob_acc'] 
        else:
            columns = ['Model','Quantile', 'QuantileLoss', 'sMAPE', 'NRMSE', 'NDRMSE','ND', 'RMSE','MAPE'] 
        quantile_data = np.round(df,3) 
        tab = PrettyTable() 
        name = self.filename+"_Quantile_analyze"
        tab.title = name 
        for i in range(1, len(columns)):
            tab.add_column(columns[i], quantile_data[:, i-1])
        tab_info = str(tab)
        print(tab_info)
        self.image_(tab_info=tab_info, imagename="Quantile_analyze")

        # 分位数为 5 的结果 
        name = self.filename+"_Analyze"
        q=2
        # ND == p50 
        sMAPE,NRMSE,NDRMSE,ND,RMSE,MAPE = quantile_data[q, 2:8] 
        p10,p50,p90,mp50 = quantile_data[0,1], quantile_data[2,1], quantile_data[4,1], round(np.mean(quantile_data[:,1]),3) 
        if self.args.eval_standard == 'prob_acc': 
            columns = ['Model','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p10','p50','p90','mp50','Prob_acc (%)'] 
            dist_acc = round(np.mean(quantile_data[q,8]),3) 
            metrics = [[self.args.model_type, sMAPE, NRMSE, NDRMSE,RMSE, MAPE, p10, p50, p90, mp50, dist_acc]]
        else:
            columns = ['Model','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p10','p50','p90','mp50'] 
            metrics = [[self.args.model_type, sMAPE, NRMSE, NDRMSE,RMSE, MAPE, p10, p50, p90, mp50]] 
        
        table = PrettyTable() 
        table.title = name 
        table.field_names = columns 
        for i in range(len(metrics)):
            table.add_row(metrics[i])                                                     # 表格内容插入 
        table_info = str(table)
        print(table_info)
        self.image_(tab_info=table_info, imagename='Quantile_5')

    def error_metrics(self, y, yhat, nrmse, ndrmse, Rp, acc=None):
        # 分位数为 5 的结果 
        name = self.filename+"_Analyze" 
        sMAPE_ = round(sMAPE(y, yhat),3)
        NRMSE_ = np.round(nrmse, 3)
        NDRMSE_ = np.round(ndrmse, 3)
        RMSE_ = round(RMSE(y, yhat),3)
        MAPE_ = round(MAPE(y, yhat),3)
        p50 = np.round(Rp, 3)
        if self.args.eval_standard == 'prob_acc':
            ACC = np.round(acc, 3) 
            columns = ['Model','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p50','Prob_acc (%)'] 
            metrics = [[self.filename, sMAPE_, NRMSE_, NDRMSE_, RMSE_, MAPE_, p50, ACC]]
        else:
            columns = ['Model','sMAPE','NRMSE','NDRMSE','RMSE','MAPE','p50'] 
            metrics = [[self.filename, sMAPE_, NRMSE_, NDRMSE_, RMSE_, MAPE_, p50]] 

        table = PrettyTable() 
        table.title = name 
        table.field_names = columns 
        for i in range(len(metrics)):
            table.add_row(metrics[i])                                                     # 表格内容插入 
        table_info = str(table)
        print(table_info) 
        self.image_(tab_info=table_info, imagename='error') 
        return sMAPE_, RMSE_, MAPE_  