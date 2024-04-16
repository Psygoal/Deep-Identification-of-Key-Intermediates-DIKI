# -*- coding: utf-8 -*-
"""
@author: Liu, XuYang
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import hdbscan

train_reonstruct_metric = keras.metrics.MeanSquaredError()
train_shrink_metric = keras.metrics.MeanSquaredError()

def init_train_step(model, optimizer):
    
    @tf.function
    def train_step(x_batch_train, y_batch_train, centers, lam=1.):
        with tf.GradientTape() as tape:
            predict, hidden, log_var = model(x_batch_train)  
            kl_loss = -0.5 * (1 + log_var - tf.square(centers-hidden) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_mean(kl_loss, axis=1))
            
            # define train loss
            train_loss = 1*keras.losses.MeanSquaredError()(y_batch_train, predict)\
                       + lam*keras.losses.MeanSquaredError()(hidden,centers)\
                       + (2. - lam)*kl_loss
            
        grads = tape.gradient(train_loss, model.trainable_weights)
    
        # BG
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        
        train_reonstruct_metric.update_state(y_batch_train, predict)
        train_shrink_metric.update_state(centers, hidden)
        
        return train_loss, kl_loss
    
    return train_step

def init_estimator(min_cluster_size=1000, min_samples=100,  cluster_selection_method='eom',core_dist_n_jobs=8):
    
    estimator = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,\
                                                            cluster_selection_method=cluster_selection_method, \
                                                            core_dist_n_jobs=core_dist_n_jobs, prediction_data=True)
    
    return estimator


def find_peaks(estimator, embeddings, sigma=0):
    
    # get information of density peaks
    labels = estimator.labels_
    
    # calculate centers of density peaks
    centers = np.vstack([np.mean(embeddings[np.where(labels==i)],axis=0,keepdims=True) for i in range(labels.max()+1)])
    
    prediction_data = estimator.prediction_data_
    core_distances = prediction_data.core_distances
    core_distances -= core_distances.min()-1e-5
    
    prob_density = 1/core_distances

    minmax_list = []
    new_labels = labels.copy()
    unique_labels = np.unique(labels)
    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    
    # estimate probability density surface
    for i in unique_labels:
        
        index = np.where(labels==i)
        
        cluster_i = prob_density[index[0]]
        cluster_i_relative = estimator.probabilities_[index[0]]
        
        min_max_idx = np.where(cluster_i_relative==1)[0]
        min_max = cluster_i[min_max_idx].min()
        
        prob_density[index[0]][min_max_idx] = min_max
        minmax_list.append(min_max)
    
    # get the discarding threshold
    discarding_threshold = np.percentile(prob_density, sigma)
    
    # discarding process
    for i in unique_labels:
        
        index = np.where(labels==i)
        
        if minmax_list[i]<discarding_threshold:
            
            new_labels[index[0]] = -1
            
    # shuffle training data
    not_outliers_idx = np.where(new_labels != -1)[0]
        
    return not_outliers_idx, centers, new_labels


def iteratively_update(all_data, model, optimizer, estimator, saved_path, sigma=0, lam=1., batch_size=128, epochs=200):
    
    train_step = init_train_step(model, optimizer)
    
    # saved metric
    loss_list = []
    
    # iteratively update the distribution of latent space
    for epoch in np.arange(epochs):
        
        # calculate embeddings
        embeddings = model(all_data)[1].numpy()
        
        # fit estimator
        estimator.fit(embeddings)
        
        # find peaks
        not_outliers_idx, centers, new_labels = find_peaks(estimator, embeddings, sigma)
        
        # define train data and the target
        train_data = all_data[not_outliers_idx]
        train_labels = new_labels[not_outliers_idx]
        
        # calculate batch steps
        batch_step = int(train_data.shape[0]/batch_size)
        
        # shuffle train data
        index = np.arange(train_data.shape[0])
        np.random.shuffle(index)
        
        # optimization
        for i in range(batch_step):
        
            # define batch data for training
            batch_index = index[i*batch_size:(i+1)*batch_size]
            
            x_batch = train_data[batch_index]
            
            labels_batch = train_labels[batch_index]
            
            centers_batch = centers[labels_batch]
            
    
            train_loss,kl_loss = train_step(tf.constant(x_batch,dtype='float32'),
                                            tf.constant(x_batch,dtype='float32'),
                                            tf.constant(centers_batch,dtype='float32'),
                                            tf.constant(lam,dtype='float32'))
            
               
            if i%5 == 0:
                print('epoch %d,batch %d \ntrain reconstruct loss: %.4f, inner loss: %.4f, kl loss: %.4f'
                  % (epoch+1, i+1, float(train_reonstruct_metric.result()),float(train_shrink_metric.result()),float(kl_loss)))
            
        # save model
        if epoch == 0:
            model.save(saved_path)
            print('model saved successfully at %s'%(saved_path))
        elif train_loss < np.min(loss_list):
            model.save(saved_path)
            print('model saved successfully at %s'%(saved_path))
            
        loss_list.append(train_loss)