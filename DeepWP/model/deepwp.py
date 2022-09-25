"""
DeepWP model 
Relevant source code: https://github.com/elephaint/pedpf/blob/master/algorithms/deepar.py 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ...modelBase import Model_ 
    from ...positionalEmbeddings import * 
except:
    from DeepWP.model.modelBase import Model_ 
    from DeepWP.model.positionalEmbeddings import *  
import os,logging,sys   

logger = logging.getLogger(__name__) 

class deepwp(Model_): 
    def __init__(self, d_lag, d_cov, d_emb, d_output, 
        d_hidden, dropout, N, dim_maxseqlen, mode, args):
        super(deepwp, self).__init__()
        self.mode = mode 
        self.args = args 
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum() 
        # Embedding layer for positions 
        self.pos_emb = SinPositionalEmbedding(d_model=d_emb_tot, max_len=dim_maxseqlen)
        # Network   
        if mode == 1:   # Elevation 
            lstm = [nn.LSTM(d_lag, d_hidden)]
        elif mode == 2 or mode == 3: # Position or Category  
            lstm = [nn.LSTM(d_lag + int(d_emb_tot), d_hidden)] 
        elif mode == 4: # Timestamp 
            lstm = [nn.LSTM(d_lag + d_cov, d_hidden)] 
        else:   # Global  
            lstm = [nn.LSTM(d_lag + d_cov + int(d_emb_tot), d_hidden)]

        for i in range(N - 1):
            lstm += [nn.LSTM(d_hidden, d_hidden)]    
        self.lstm = nn.ModuleList(lstm)
        self.drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N)])
        # Output distribution layers 
        self.loc = nn.Linear(d_hidden * N, d_output)
        self.scale = nn.Linear(d_hidden * N, d_output) 
        self.epsilon = 1e-6

        # Number of parameters 
        total_params, total_trainable_params = self.num_parameters
        logger.info(f'{total_params:,} total parameters.')
        logger.info(f'{total_trainable_params:,} training parameters.') 

    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen, mode=None):
        """
        Args:
            x_lag (torch.tensor) : (n_lag+n_seq, batch_size, d_lag)
            x_cov (torch.tensor) : (n_lag+n_seq, batch_size, d_cov)
            x_idx (torch.tensor) : (n_lag+n_seq, batch_size, d_emb) 
            d_outputseqlen (torch.tensor) : (n_seq) 
        Returns:
            loc (torch.tensor)   : (n_seq, batch_size, d_output)
            scale (torch.tensor) : (n_seq, batch_size, d_output) 
        """ 
        # Embedding layers
        emb_id = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])                     # (n_lag+n_seq, batch_size, emd_dim)
            emb_id.append(out)
        emb_id = torch.cat(emb_id, -1)[:x_lag.shape[0]]
        # position embedding 
        pos_emb = self.pos_emb(x_lag.permute(1, 0, 2)).permute(1, 0, 2) # (n_lag+n_seq, 1, d_emb_tot) | (n_lag+n_seq, batch_size, d_emb_tot)
        if pos_emb.shape[1] == 1:
            pos_emb_re = pos_emb.repeat(1, x_lag.shape[1], 1)   # (n_lag+n_seq, batch_size, d_emb_tot)
            x_emb = emb_id +  pos_emb_re                        # (n_lag+n_seq, batch_size, d_emb_tot)
        elif pos_emb.shape[1] == x_lag.shape[1]:
            x_emb = emb_id +  pos_emb
        else:
            print("Error in deepwp.py Line")
            os.system("pause")
            sys.exit(1)
        # Concatenate x_lag, x_cov and time series ID
        dim_seq = x_lag.shape[0]
        # (n_lag+n_seq, batch_size, d_lag+d_cov+d_emb) 
        if self.mode == 1:      
            inputs = x_lag 
        elif self.mode == 2:   
            if pos_emb.shape[1] == 1:
                inputs = torch.cat((x_lag, pos_emb_re[:dim_seq]), dim=-1)   
            elif pos_emb.shape[1] == x_lag.shape[1]:
                inputs = torch.cat((x_lag, pos_emb[:dim_seq]), dim=-1)   
            else:
                print("Error in deepwp.py Line")
                os.system("pause")
                sys.exit(1)
        elif self.mode == 3:    
            inputs = torch.cat((x_lag, emb_id[:dim_seq]), dim=-1) 
        elif self.mode == 4:    
            inputs = torch.cat((x_lag, x_cov[:dim_seq]), dim=-1) 
        else:                   
            inputs = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1) 
        
        # DeepWP network       
        h = []
        for i, layer in enumerate(self.lstm):
            outputs, _ = layer(inputs)                      # (n_lag+n_seq, batch_size, d_hidden)
            outputs = self.drop[i](outputs)
            inputs = outputs
            h.append(outputs)
        h = torch.cat(h, -1)                                # (n_lag+n_seq, batch_size, d_hidden * N)
        # Output layers - location and scale of distribution
        loc = self.loc(h[-d_outputseqlen:])                 # (n_seq, batch_size, d_output)
        scale = F.softplus(self.scale(h[-d_outputseqlen:])) # (n_seq, batch_size, d_output) 
        return loc, scale + self.epsilon 

if __name__ == '__main__': 
    # import numpy as np 
    # d_emb = np.array([[370, 20]])
    # model = deepwp(d_lag=1, d_cov=7, d_emb=d_emb, d_output=1, d_hidden=40, dropout=0.1, N=3, 
    # dim_maxseqlen=12, mode=2,args=None) 
    # print(model)
    # os.system("pause")
    # # print(model.pos_emb.pe.shape)
    # # print(model.pos_emb.div_term.data)
    # # os.system("pause")
    # # sys.exit(1)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #         print(param.shape)
    #         os.system("pause")
    # x_lag = torch.rand(192, 32, 1) 
    # x_cov = torch.rand(500, 32, 7) 
    # x_idx = torch.randint(low=1, high=4, size=(500, 32, 1))
    # loc, scale = model(x_lag, x_cov, x_idx, 24)
    # print(loc.shape)
    # print(scale.shape)
    # os.system("pause")
    pass 