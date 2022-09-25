import matplotlib.pyplot as plt 
import os, logging, sys 
import numpy as np 
import datetime, time 
from prettytable import PrettyTable 
from PIL import Image, ImageDraw 

class ALLDataAnalyze(object):
    """ 
    Parent class for analyzing all result datas 
    """
    def __init__(self, args=None, dir=None, filename=None, dpi=100, fig_extension="png", save=None, plt_show=None):
        self.args = args 
        self.dir = dir 
        self.dpi = dpi 
        self.fig_extension = fig_extension 
        self.filename = filename  
        if save is None:
            self.save = args.save 
        else:
            self.save = save
        if plt_show is None:
            self.plt_show = args.plt_show
        else: 
            self.plt_show = plt_show 

    def save_fig(self, filename):
        if os.path.isfile(self.dir):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.dir))
        os.makedirs(name=self.dir, exist_ok=True)

        path = os.path.join(self.dir, filename + "." + self.fig_extension)
        plt.tight_layout()  # 调整图片的布局 
        plt.savefig(path, format=self.fig_extension)
        if self.plt_show:
            plt.show()          # by liuhahayue 
        else:
            plt.close()         # 将图片直接关闭不需要展示 

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
        if self.args.save:
            self.save_image(img=im_new, filename=self.filename)
        # im_new.show()
        del draw 

    # Result analyze 
    def save_image(self, img, filename):
        if os.path.isfile(self.dir):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(self.dir))
        os.makedirs(name=self.dir, exist_ok=True)
        path = os.path.join(self.dir, filename + ".png") 
        img.save(path) 

    def result2csv(self, df, mode=None):
        if mode == 'kflod':
            filesuffix = f"{self.args.model_type}_KFlod_analyze"
        else:
            # to .csv format file 
            if self.args.model_configpath is not None: 
                _, config_file = os.path.split(self.args.model_configpath) 
                filesuffix, _ = os.path.splitext(config_file) 
            else:
                curT = datetime.datetime.now() 
                filesuffix = f"{curT.year}_{curT.month}_{curT.day}_{curT.hour}_{curT.minute}_{curT.minute}" 
        self.filename = f"{filesuffix}_src={self.args.data_source}" 
        result_filepath = os.path.join(self.dir, self.filename+'.csv') 
        df.to_csv(result_filepath, index=False) 
        result_table = PrettyTable() 
        if mode == 'kflod':
            result_table.title = f"All K-Flods Results Analyze Quantile 5" 
        else:
            result_table.title = f"All Results Analyze Quantile 5" 
        result_table.field_names = df.columns  
        for i in range(df.shape[0]):
            result_table.add_row(df.iloc[i, :]) 

        table_info = str(result_table) 
        print(table_info) 
        self.image_(tab_info=table_info, imagename='') 

    def get_yticks_range(self, y, mode):
        max_val = np.max(np.concatenate(y)) 
        min_val = np.min(np.concatenate(y)) 
        rp_type = ['rp1', 'rp5', 'rp9', 'mp', 'nrmse', 'ndrmse']
        if mode in rp_type:
            p001 = max_val * 1000 % 10      # 0.001 分位 
            p01 = max_val * 100 % 10 // 1   # 0.01 分位 
            p1 = max_val * 10 // 1          # 0.1 分位 
            if p001 >= 5 :
                if p01 < 9:
                    max_val = round((p1*100 + (p01+1)*10)*0.001, 3)
                else:
                    max_val = round((p1+1)*0.1, 3)
            else:
                max_val = round((p1*100 + p01*10 + 5)*0.001, 3) 

            p001 = min_val * 1000 % 10      # 0.001 分位 
            p01 = min_val * 100 % 10 // 1   # 0.01 分位 
            p1 = min_val * 10 // 1          # 0.1 分位 
            if p001 >= 5:
                min_val = round((p1*100 + p01*10 + 5)*0.001, 3) 
            else:
                min_val = round((p1*100 + p01*10)*0.001, 3)
            stride = (max_val - min_val) / 5 
            yticks = np.arange(min_val, max_val+0.001, stride) 
            if mode == 'ndrmse':
                y_label = 'NDRMSE' 
            elif mode == 'rp1':
                y_label = 'Rp1'
            elif mode == 'rp5':
                y_label = 'Rp5' 
            elif mode == 'rp9':
                y_label = 'Rp9' 
            elif mode == 'mp':
                y_label = 'M_Rp'
            elif mode == 'nrmse':
                y_label = 'NRMSE'
        elif mode == 'acc':
            p10 = max_val // 10                 # 10 分位 
            max_val = round((p10+1)*10, 3) 

            if min_val >= 10:
                p10 = min_val // 10                 # 10 分位 
                min_val = round(p10*10, 3) 
            else:
                min_val = round(0, 3)
            yticks = np.arange(min_val, max_val+1, 10)
            y_label = 'Acc'
        elif mode == 'etime':
            p1 = max_val // 1                   # 个位 
            max_val = round(p1+1, 3)
            p1 = min_val // 1 
            min_val = round(p1-1, 3)
            yticks = np.arange(min_val, max_val+1, 1) 
            y_label = 'Epoch Time (s)' 
        elif mode == 'para' or mode == 'memory' or mode == 'bpw' or mode == 'tdp' or mode == 'time':
            if max_val // min_val > 8:
                stride = (max_val - min_val) // 8 
                yticks = np.arange(min_val-1, max_val+1, stride) 
            else:
                stride = (max_val - min_val) // 5 
                yticks = np.arange(min_val-1, max_val+1, stride) 
        if mode == 'para':
            y_label = 'Parameters'
        elif mode == 'memory':
            y_label = 'Memory (MB)'
        elif mode == 'bpw':
            y_label = 'Board Power Draw (W)' 
        elif mode == 'tdp':
            y_label = 'Power Consumption (%) ' 
        elif mode == 'time':
            y_label = 'Train Time (s) ' 
        return yticks, y_label  
    
    def plot_differnet_covariate(self, results, mode, legends, titles, calmode, n_seq=None, split=None): 
        """
        Args:
            results (np.array) : (n_covariate, n_modeltype, times) (协变量类型， 模型， 。。。)
            mode : 待计算变量名(纵轴标识) | legends : 右上角标识 | titles : 子标题 
            calmode : 计算模式 
        """ 
        n_covariate, n_modeltype, times = results.shape 
        # 坐标刻度 
        xticks = []
        for time in range(times):
            xticks.append(int(n_seq)*(time+1)) 
        marker = ["o", "s", "*", "D", "H", "p", "v", "1", "X", "2", "3", "4"]
        # legend 位置 
        ul = ['etime', 'para', 'memory', 'bpw', 'tdp']
        rp_type = ['rp1', 'rp5', 'rp9', 'mp', 'nrmse', 'ndrmse', 'time']
        if mode in rp_type:
            loc = 'upper right'
        elif mode == 'acc':
            loc = 'lower right' 
        elif mode in ul :
            loc = 'upper left'
        else:
            loc = 'center right'
        if n_covariate == 1:
            ax = plt.gca()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=n_covariate, figsize=(n_covariate*2.5,3), sharey='row')
        x = np.arange(n_seq, n_seq*times+1, n_seq) 
        models_covariates = [] 
        for i in range(n_covariate):    # 循环遍历 协变量  
            if n_covariate == 1:
                ax_sub = ax 
            else:
                ax_sub = ax[i]
            for j in range(n_modeltype):    # 循环遍历 模型类型  
                if calmode == 'avg':
                    if split:
                        avg = np.mean(results[i][j][:split])    # (8,) 
                    else:
                        avg = np.mean(results[i][j]) 
                elif calmode == 'min':
                    avg = np.min(results[i][j]) 
                else:
                    print("You should check the code Line 239") 
                    os.system("pause")
                    sys.exit(1)
                if mode == 'para':
                    sub_label = f"{legends[j]} ({calmode}={int(avg):d})"
                else:
                    sub_label = f"{legends[j]} ({calmode}={avg:.3f})"
                ax_sub.plot(x, results[i][j], marker=marker[j], markersize=4, label=sub_label)
                models_covariates.append(results[i][j])
            
            ax_sub.set_title(label = titles[i]) # 子标题 
            ax_sub.set_xticks(xticks)    # x轴刻度 
            ax_sub.legend(loc=loc, fontsize=7)  # color bar 
            ax_sub.set_xlabel('Input length') # x轴标识 
        # Y 坐标刻度 
        yticks, y_label = self.get_yticks_range(models_covariates, mode) 
        if n_covariate == 1:
            ax.set_yticks(yticks)    # y轴刻度 
            ax.set_ylabel(y_label)   # y轴标识 
        else:
            ax[0].set_yticks(yticks)    # y轴刻度 
            ax[0].set_ylabel(y_label)   # y轴标识 
        # Save figure 
        name = f"{self.filename} ({y_label}_{calmode})"     # 总标题 
        plt.suptitle(name) 
        if self.save:
            self.save_fig(name)