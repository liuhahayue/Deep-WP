# Deep-WP
Deep Wave Prediction (Deep-WP) is a repository containing code to run examples in our paper : 
[Deterministic wave prediction model for irregular long-crested waves with Recurrent Neural Network](https://www.sciencedirect.com/science/article/pii/S2468013322002340).

## Deep Learning Wave Prediction Model
Real-time predicting of stochastic waves is crucial in marine engineering. In this paper, a deep learning wave prediction (Deep-WP) model based on the ‘**probabilistic**’ strategy is designed for the short-term prediction of stochastic waves. The Deep-WP model employs the long short-term memory (LSTM) unit to collect pertinent information from the wave elevation time series. Five irregular long-crested waves generated in the deepwater offshore basin at Shanghai Jiao Tong University are used to validate and optimize the Deep-WP model. When the prediction duration is **1.92s**, **2.56s**, and, **3.84s**, respectively, the predicted results are almost identical with the ground truth. As the prediction duration is increased to **7.68s** or **15.36s**, the Deep-WP model’s error increases, but it still maintains a high level of accuracy during the first few seconds. The introduction of covariates will improve the Deep-WP model’s performance, with the **absolute position** and **timestamp** being particularly advantageous for wave prediction. Furthermore, the Deep-WP model is applicable to predict waves with different energy components. The proposed Deep-WP model shows a feasible ability to predict nonlinear stochastic waves in real-time.

## <span id="citelink">Citation</span> 
If you find this repository useful in your research, please consider citing the following paper:

```
@article{LIU-DeepWP-2022,
title = {Deterministic wave prediction model for irregular long-crested waves with Recurrent Neural Network},
journal = {Journal of Ocean Engineering and Science},
year = {2022},
issn = {2468-0133},
doi = {https://doi.org/10.1016/j.joes.2022.08.002},
url = {https://www.sciencedirect.com/science/article/pii/S2468013322002340},
author = {Yue Liu and 
          Xiantao Zhang and 
          Gang Chen and 
          Qing Dong and 
          Xiaoxian Guo and 
          Xinliang Tian and 
          Wenyue Lu and 
          Tao Peng},
}
```
