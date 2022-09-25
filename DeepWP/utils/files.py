"""
2021.8.10 by liuhahayue
"""
import os, logging, sys  
from pathlib import Path 

logger = logging.getLogger(__name__)

def pathcheck(dirname):
    cur_filepath = sys.path[0]
    main_dir = os.path.join(cur_filepath, dirname)
    return main_dir 

def filecheck(filename):
    filepath = os.path.join(sys.path[0], filename) 
    if not(os.path.isfile(filepath)):
        prev_path = os.path.abspath(os.path.join(sys.path[0],"..")) 
        filepath = os.path.join(prev_path, filename) 
        if not(os.path.isfile(filepath)):
            logger.info(f"The file {filename} is not found !!! ") 
            os.system("pause")
            sys.exit(1)
    return filepath 

def config_filescheck(dir, prefix=None):
    direcotry = Path(os.path.join(sys.path[0], dir)) 
    if not(os.path.exists(path=direcotry)):
        prev_path = os.path.abspath(os.path.join(sys.path[0],"..")) 
        direcotry = Path(os.path.join(prev_path, dir)) 
        if not(os.path.exists(direcotry)):
            logger.info(f"The config {dir} is not found")
            os.system("pause") 
            sys.exit(1)
    if prefix is None:
        configFlie = "*.csv" 
    else:
        configFlie = f"{prefix}*.csv" 
    configFiles = list(direcotry.glob(configFlie)) 
    if configFiles:
        return configFiles 
    else:
        logger.info(f" The directory {direcotry} is null") 
        os.system("pause") 
        sys.exit(1)