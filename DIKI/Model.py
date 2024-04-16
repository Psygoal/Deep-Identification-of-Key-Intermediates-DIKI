# -*- coding: utf-8 -*-
"""
@author: Liu, XuYang
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

class Sampling_Layer(keras.layers.Layer):
    def call(self,inputs):
        mean, log_var = inputs
        random = tf.keras.backend.random_normal(shape=(tf.shape(mean)[0],tf.shape(mean)[1]))
        sampling_results = mean + tf.exp(0.5*log_var)*random
        kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(1e-2*kl_loss)
        return sampling_results
    

def VAEmodel(feature_number, layers_dim, latent_dim=2, activation='leaky_relu', \
             kernel_initializer=keras.initializers.HeNormal()):
    """
    build the VAE model
    ------------------------------Input------------------------------
    
    feature_number: int, a list or array to initialize the input_shape of neural network. 
                    e.g., for Cartesian Coordinates, it should be a two-dimensional list, like [93,3]
                    
    layers_dim: int, a list of array of output dimension of each Dense layer, e.g., [500,400,300,200]
    
    latent_dim: int, dimension of the latent space
    
    activation: str or keras object, activation function of each Dense layer
    
    kernel_initializer: str or keras object, initialization way of each Dense layer
    
    ------------------------------Output------------------------------
    
    model: keras obeject, VAE model
    
    """
    
    inputs = keras.layers.Input(shape=feature_number)
    
    # if the feature_number is 2-dimensional
    outputs = keras.layers.Flatten()(inputs)
    
    for dim in sorted(layers_dim, reverse=True):
        outputs = keras.layers.Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(outputs)
    
    encoded = keras.layers.Dense(latent_dim,kernel_initializer=keras.initializers.HeNormal(),name='encoded')(outputs)
    # 'tanh' for keep continuity in the latent space, or it will be too negative and leads to VAE degeneration
    log_var = keras.layers.Dense(latent_dim,activation='tanh',kernel_initializer=keras.initializers.HeNormal(),name='log_var')(outputs)  
    sampling = Sampling_Layer()([encoded,log_var])
    
    outputs = sampling
    
    for dim in sorted(layers_dim):
        outputs = keras.layers.Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(outputs)

    outputs = keras.layers.Dense(np.prod(feature_number),kernel_initializer=keras.initializers.HeNormal())(outputs)
    outputs = keras.layers.Reshape(feature_number,name='reconstruction')(outputs)
    
    model = keras.models.Model(inputs,[outputs,encoded,log_var])
        
    return model



