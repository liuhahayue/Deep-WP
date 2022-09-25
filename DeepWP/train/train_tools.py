from collections import defaultdict, OrderedDict  
from pathlib import Path 
import torch 
import os, sys 

class Checkpoints():
    def __init__(self, filepath, filename, to_keep=10):
        # initialize the directory path and the directory 
        self.directory = Path(filepath)
        self.filename = filename 
        self.to_keep = to_keep 
        if len(list(self.directory.glob('%s*' % self.filename))) > 0:
            for model_file in list(self.directory.glob('%s*' % self.filename)):
                os.remove(model_file)

    def save(self, env, epoch):
        checkpoints = list(self.directory.glob('%s*' % self.filename))
        if len(checkpoints) > self.to_keep:
            checkpoints[0].unlink()
        checkpoint_name = f"{self.filename}_{epoch}.pth" 
        torch.save(env.model.state_dict(), self.directory/checkpoint_name)      
        return self.directory/checkpoint_name   

class Validation():
    def __init__(self, filepath=None, filename=None, keep_best=5, checkpoint=True):
        self.metric_log = defaultdict(list)
        self.checkpoint = checkpoint 
        self.filepath = filepath 
        self.filename = filename 
        if checkpoint:
            self.checkpointer = Checkpoints(filepath, filename)
            if keep_best:
                assert isinstance(keep_best, int), "keep_best must be an integer" 
                self.keep_best = keep_best
                self.best_scores = OrderedDict() 
                self.checkpointer.to_keep = keep_best 

    def save_best_min(self, metrics, env, epoch):
        self.metric_log['accuracy'].append(metrics) 
        if not self.best_scores:    
            self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1]
        else:
            if len(self.best_scores) == self.keep_best: 
                for key in self.best_scores.keys():
                    if self.metric_log['accuracy'][-1] < self.best_scores[key]:
                        if key.exists():
                            os.remove(key.as_posix()) 
                            self.best_scores.pop(key)
                            self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1]
                            break 
                self.sort_best_scores_min()     
            else:
                self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1] 
                self.sort_best_scores_min()     

    def sort_best_scores_min(self):
        tmp = [(a, b) for a, b in self.best_scores.items()] 
        tmp.sort(key=lambda x: x[1], reverse=True)
        self.best_scores = OrderedDict(tmp)

    def save_best(self, metrics, env, epoch):
        self.metric_log['accuracy'].append(metrics)
        if not self.best_scores:
            if self.metric_log['accuracy'][-1] <= 100:
                if self.metric_log['accuracy'][-1] > 0: 
                    self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1]
        else:
            if len(self.best_scores) == self.keep_best: 
                for key in self.best_scores.keys(): 
                    if self.metric_log['accuracy'][-1] > self.best_scores[key]: 
                        if self.metric_log['accuracy'][-1] < 100: 
                            if key.exists(): 
                                os.remove(key.as_posix())   
                                self.best_scores.pop(key)   
                                self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1] 
                                break 
                self.sort_best_scores()     
            else:
                self.best_scores[self.checkpointer.save(env, epoch)] = self.metric_log['accuracy'][-1] 
                self.sort_best_scores()     

    def sort_best_scores(self):
        tmp = [(a, b) for a, b in self.best_scores.items()]
        tmp.sort(key=lambda x: x[1])
        self.best_scores = OrderedDict(tmp)