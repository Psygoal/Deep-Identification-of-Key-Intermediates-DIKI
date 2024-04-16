![header](images/toc.jpg)
# DIKI
Deep Identification of Key Intermediates

DIKI is a VAE-based neural network for analyzing conformations in Molecular Dynamics trajectories.
The work has been accpeted by Journal of Chemical Theory and Computation (JCTC). Please cite our 
paper if you use DIKI in your work:  
  
&ensp;&ensp; X. Liu, J. Xing, W. Cai, X. Shao. Analyzing Molecular Dynamics Trajectories Thermodynamically through Artificial Intelligence. Journal of Chemical Theory and Computation, 2024, xxx: xxx-xxx.  

We provide the following codes for use:  
&ensp;&ensp; 1. align a dcd trajectory to a reference structure and output a npz file of aligned Cartesian coordinates.  
&ensp;&ensp; 2. build a VAE model  
&ensp;&ensp; 3. use HDBSCAN iteratively updating the latent space  
&ensp;&ensp; 4. plot the similarity heatmap used in our paper  

## Virtual Environment  
The required packages and their versions are included in requirement.txt file. Run the following commands to build your environment:  
```bash
conda create -n DIKI python=3.8.11  
conda activate DIKI  
pip install -r requirement.txt
```

## Use of DIKI
The commonly-used hyperparameters of DIKI is defined in ```parameters.json```. Users can simply run
```bash
python ./DIKI/main.py -p ./parameters.json
```
to training DIKI.

## Parameters interpretation
&ensp;&ensp;```is_aligned``` bool, 0 or 1. Whether the dcd trajectory is aligned. If the value is set as 0, the Cartesian coordinates matrix will aligned with the frame-0 structure, and the aligned results will saved at ```aligned_npz```, otherwise the coordinates will be loaded from ```aligned_npz```.  
<br/> 
&ensp;&ensp;```is_warmedup``` bool, 0 or 1. Whether the VAE model is pretrained. If the value is set as 0, the model will be initiliazed and trained from the beginning, and the trained model will saved at ```warmed_up_model_path```, otherwise the model will be loaded from ```warmed_up_model_path```.   
<br/> 
