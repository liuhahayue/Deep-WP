import logging, os, sys  
from DeepWP.train import MultiExperiments 
from DeepWP.utils import get_args, pathcheck, filecheck  

if __name__ == "__main__": 
    logger = logging.getLogger(__name__) 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO) 

    # ================= Initialize Args Config 
    args = get_args() 

    # ================= Basic Configuration 
    args.main_dir = pathcheck('main')                  
    args.save = True  
    args.plt_show = False                                
    args.forecast_type = 'probability'                  
    # Debug Args: 
    args.debug = True                                    

    # ================= Dataset 
    args.filename = "dataset\WC04-1-2.5-Full.dat"      
    args.filepath = filecheck(args.filename)
    args.fileformat = 1                                 
    args.data_source = "EXP"                            
    args.Hs = 12.0 
    args.delta_t = 320                                  
    args.remove_time = 180                              
    args.timespan = 4096                                
    args.location = [0] 
    args.scaling_train = True                       
    args.n_covariate = 1
    # Debug Args: 
    args.debug_id = 0 

    # ================= Model Basic Configuration 
    args.model_configfile = "config\deepwp_6.csv" 
    args.model_configpath = filecheck(args.model_configfile)
    args.n_modeltype = 5 

    # ================= Training Hyperparameters 
    # Basic Args: 
    args.epochs = 1                                     
    args.eval_standard = 'ndrmse'                          
    args.early_stop = 20                                
    # Optimizer Args: 
    args.optimizer_mode = 1                             
    args.weightDecay = 1e-8  
    args.scheduler_dynamic = True  
    # Probability Class Args: 
    args.pdf = 'gaussian' 
                                  
    # ================= Testing Hyperparameters 
    args.test_number = 0 
    args.test_all = True                               
    
    # ================= Initialize Experiments 
    experiment = MultiExperiments(args) 

    # Train and Test 
    experiment.LoopModels()