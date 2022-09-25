import pickle
import os, logging 
import json
import pandas as pd
import numpy as np 
import time 
from filelock import FileLock 

logger = logging.getLogger(__name__) 

def save(obj, filename):
    filename += ".pickle" if ".pickle" not in filename else ""
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def default(o):
    if isinstance(o, np.int64):
        return int(o)

def save_json(dict, filename):
    with open(filename, 'w') as fp:
        json.dump(dict, fp, default=default)
    fp.close()

def load_dat(filepath, mode):
    """
    Args:
        filepath (str): Cache file path
        mode (int): mode=1 水池波浪数据 mode=2 水池运动响应数据 
    Return:
        examples (np.array): data (ndata, seq_len)
    """
    assert os.path.isfile(filepath), 'Provided cache file path does not exist!' 
    start = time.time() 
    lock_path = str(filepath)+ ".lock" 
    with FileLock(lock_path): 
        if mode == 1:
            wave_df = pd.read_csv(
                filepath_or_buffer=filepath,  engine='python', sep = '\s+'
            )
            wave_df.drop(index=[0], inplace=True)
            waveDatas = []
            for column in wave_df: 
                temp_data = wave_df[column]
                temp_data = list(map(np.float32, temp_data))    # (str 2 float)
                waveDatas.append(temp_data)
            # list 2 np.array 
            waveDatas = np.array(waveDatas) 
            logger.info(
                f"Loading features from cached file {filepath} [took %.3f s]", time.time() - start)
            return waveDatas[1:-1] 
        elif mode == 2:
            wave_df = pd.read_csv(
                filepath_or_buffer=filepath, header=None, engine='python', sep = ' '
            )
            waveDatas = []
            for column in wave_df: 
                temp_data = wave_df[column] 
                temp_data = list(map(np.float32, temp_data))    # (str 2 float)
                waveDatas.append(temp_data)
            # list 2 np.array 
            waveDatas = np.array(waveDatas) 
            logger.info(
                f"Loading features from cached file {filepath} [took %.3f s]", time.time() - start)
            return waveDatas[1:] 