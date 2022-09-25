import os, logging, sys  
import torch 
import torch.nn as nn 
from abc import abstractmethod 

logger = logging.getLogger(__name__)

class Model_(nn.Module):
    """
    Parent class for every models 
    """
    def __init__(self):
        super(Model_, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward function has not been properly overridden")

    @property
    def num_parameters(self):
        """
        Get number of learnable parameters in model
        """
        # Find total parameters and trainable parameters 
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, total_trainable_params 

    @property
    def devices(self):
        """Get list of unique device(s) model exists on
        """
        devices = []
        for param in self.parameters():
            if(not param.device in devices):
                devices.append(param.device)
        for buffer in self.buffers():
            if (not buffer.device in devices):
                devices.append(buffer.device)
        return devices

    def save_model(self, name, save_directory: str):
        """Saves model to the specified directory.

        Args:
            save_directory (str): Folder directory to save state dictionary to.
            
        Raises:
            FileNotFoundError: If provided path is a file
        """
        if os.path.isfile(save_directory):
            raise FileNotFoundError("Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        output_model_file = os.path.join(save_directory, "{}.pth".format(name))
        # Save pytorch model to file
        logger.info('Saving the best testing model file: {}'.format(output_model_file))
        torch.save(self.state_dict(), output_model_file)   

    def load_model(self, name, file_or_path_directory: str):
        """Load a testing model from the specified file or path
        
        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
        
        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading the best testing model from file: {}'.format(file_or_path_directory))
            self.load_state_dict(torch.load(file_or_path_directory, map_location=lambda storage, loc: storage))
        elif  os.path.isdir(file_or_path_directory):
            file_path = os.path.join(file_or_path_directory, "{}.pth".format(name))
            logger.info('Loading the best testing model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        else:
            raise FileNotFoundError("Provided path or file ({}) does not exist".format(file_or_path_directory))