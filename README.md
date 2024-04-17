![header](images/toc.jpg)
# DIKI
Deep Identification of Key Intermediates

DIKI is a VAE-based neural network for analyzing conformations in Molecular Dynamics trajectories.
The work has been published on Journal of Chemical Theory and Computation (JCTC). Please cite our 
paper:  
  
&ensp;&ensp; X. Liu, J. Xing, W. Cai, X. Shao. Analyzing Molecular Dynamics Trajectories Thermodynamically through Artificial Intelligence. Journal of Chemical Theory and Computation, 2024, 20(2): 665-676. DOI: [10.1021/acs.jctc.3c00975  ](http://dx.doi.org/10.1021/acs.jctc.3c00975)

We provide the following codes for use:  
&ensp;&ensp; 1. align a dcd trajectory to a reference structure and output a npz file of aligned Cartesian coordinates.  
&ensp;&ensp; 2. build a VAE model  
&ensp;&ensp; 3. use HDBSCAN iteratively updating the latent space  
&ensp;&ensp; 4. plot the similarity heatmap used in our paper  

## Virtual environment  
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
&ensp;&ensp;```psf_file_path``` str. The path of tology file.    
<br/> 
&ensp;&ensp;```dcd_file_path``` str. The path of trajectory file.    
<br/> 
&ensp;&ensp;```aligned_npz``` str. The path of aligned coordinates, saved as npz file.    
<br/> 
&ensp;&ensp;```selection``` str. Atom selection language. Details can be found in MDanalysis docs.   
<br/>
&ensp;&ensp;```warmed_up_model_path``` str. The path of warmed_up model, saved as h5 format.   
<br/>
&ensp;&ensp;```min_cluster_size``` int. A pararmter of HDBSCAN, indicating the minimum size of a cluster, also called CFK in our paper.   
<br/>
&ensp;&ensp;```min_samples``` int. A pararmter of HDBSCAN, indicating the number of samples to calculate core distances, also called FFK in our paper.       
<br/>
&ensp;&ensp;```cluster_selection_method``` str. A pararmter of HDBSCAN, which can be set as "eom" or "leaf".    
<br/>
&ensp;&ensp;```sigma``` int. The percentile of probability density, playing a role of threshold for discarding high-free-energy clusters.    
<br/>
&ensp;&ensp;```DIKI_saved_path``` str. The path of DIKI model, saved as h5 format.     
<br/>
&ensp;&ensp;```lam``` float. The weight of KL loss, and ```2-lam``` indicates the weight of shrinking loss.    
<br/>
&ensp;&ensp;```DIKI_batch_size``` int. Batch size of iterative update of DIKI.    
<br/>
&ensp;&ensp;```DIKI_epochs``` int. Epochs of iterative update of DIKI.     
<br/>
&ensp;&ensp;```encoding_info_path``` str. The path of DIKI encodings and clustering results, saved as csv format.     
<br/>
&ensp;&ensp;```similarity_heatmap_saved_path``` str. The path of similarity heatmap, saved as jpg format.     
