# -*- coding: utf-8 -*-
"""
@author: LXY
"""

from DataPreprocessing import alignDcd2Ref
from Model import VAEmodel
from PosteriorReplan import init_estimator,iteratively_update
from Model import Sampling_Layer
from SimilarityPlot import simliarityplot

from tensorflow import keras
import json
import numpy as np
import pandas as pd
import tensorflow as tf

## parameters of alignDcd2Ref
import argparse



if __name__ == '__main__':
    # hyperparameters
    parser = argparse.ArgumentParser(description='hyperparameters')
    parser.add_argument('-p','--parameters',type=str, help='path of the JSON file of parameters')
    args = parser.parse_args()
    parameter_file=args.parameters

    f = open(parameter_file,'r',encoding='utf-8')
    m = json.load(f)
    for key in list(m.keys()):
        exec('%s = m["%s"]'%(key,key))
    
    # align dcd 2 reference
    if is_aligned:
        npz = np.load(aligned_npz)
        all_data = npz['position']
        npz.close()
        
    else:
        all_data = alignDcd2Ref(psf_file_path, dcd_file_path, aligned_npz, selection)

    # warm-up stage
    ## load or train warm-up model
    if is_warmedup:
        
        model = keras.models.load_model(warmed_up_model_path, custom_objects={'Sampling_Layer':Sampling_Layer})
        
    else:
        no_use = tf.zeros((tf.shape(all_data)[0],2))
        feature_number = all_data.shape[1:]
        model = VAEmodel(feature_number,layers_dim)
        
        warm_up_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss=['mse','mse','mse'],
                      loss_weights=[1,0,0],
                      optimizer=warm_up_optimizer)
        
        ## model training
        earlystop = keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=50, mode='auto')

        savebestmodel = keras.callbacks.ModelCheckpoint(warmed_up_model_path, 
                                                        monitor = 'reconstruction_loss', 
                                                        verbose = 1,                                                      
                                                        save_best_only = True, 
                                                        mode = 'auto')
        warm_up_history = model.fit(all_data,[all_data,no_use,no_use],
                                     epochs=3000,
                                     batch_size=256,
                                     callbacks=[earlystop,savebestmodel])
    
    
    # iteratively update
    estimator = init_estimator(min_cluster_size,min_samples,cluster_selection_method)
    
    # Replan distribution
    iter_update_optimizer = keras.optimizers.RMSprop(learning_rate=1e-4)
    iteratively_update(all_data, model, iter_update_optimizer, estimator, DIKI_saved_path, sigma=sigma, lam=lam, batch_size=DIKI_batch_size, epochs=DIKI_epochs)
    
    # write encoding informations
    embeddings = model(all_data)[1].numpy()
    estimator.fit(embeddings)
    df = pd.DataFrame()
    df['frames'] = np.arange(1,len(all_data)+1)
    df['encoding1'] = embeddings[:,0]
    df['encoding2'] = embeddings[:,1]
    df['labels'] = estimator.labels_
    df.to_csv(encoding_info_path,index=False)
    
    # plot similarity figure
    simliarityplot(model,estimator,all_data,similarity_heatmap_saved_path)
