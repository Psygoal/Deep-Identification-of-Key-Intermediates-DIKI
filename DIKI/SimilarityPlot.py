# -*- coding: utf-8 -*-
"""
@author: LXY
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch

def simliarityplot(model, estimator, Carte_coor, img_saved_path='similarity_heatmap.png', num=500):
    
    embeddings = model(Carte_coor)[1].numpy()
    estimator.fit(embeddings)
    labels = estimator.labels_
    max_label = labels.max()
    
    sim_mat = []
    # calculate cosine similarity
    for i in range(max_label+1):
        
        idx = np.where(labels==i)[0]
        
        center = np.mean(embeddings[idx],keepdims=True,axis=0)
        
        distance = np.sum((embeddings[idx] - center)**(2),axis=1)
        
        min_idx = np.argsort(distance)[:num]
        
        sim_mat.append((Carte_coor[idx][min_idx]).reshape((num,-1)))
        
    sim_mat = np.vstack(sim_mat)
    
    sim_mat /= np.linalg.norm(sim_mat,ord=2,axis=1,keepdims=True)
    SIMILARITY = sim_mat@(sim_mat.T)
    x_y_coor_ = np.linspace(num, (max_label+1)*num, max_label+1)
    
    # cutoff outliers
    cbar_ub = np.percentile(SIMILARITY.flatten(),95)
    cbar_lb = np.percentile(SIMILARITY.flatten(),5)
    
    # plot figure
    plt.figure(figsize=[7,6])
    plt.imshow(SIMILARITY,cmap='jet')
    plt.clim(cbar_lb,cbar_ub)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(np.linspace(num*0.5,num*(max_label+0.5),max_label+1),['C%d'%(i+1) for i in range(max_label+1)],fontsize=20)
    plt.yticks(np.linspace(num*0.5,num*(max_label+0.5),max_label+1),['C%d'%(i+1) for i in range(max_label+1)],fontsize=20)
    ax=plt.gca()
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    
    # add patches
    for i in range(max_label+1):
        if i == 0:
            rct = pch.Rectangle((0,0),num,num,edgecolor='k',lw=3,facecolor='None')
            ax.add_patch(rct)
        else:
            rct = pch.Rectangle((x_y_coor_[i-1]+0,x_y_coor_[i-1]+0),num,num,edgecolor='k',lw=3,facecolor='None')
            ax.add_patch(rct)
    
    # save figure
    plt.savefig(img_saved_path,bbox_inches='tight',pad_inches=0.1,dpi=300)