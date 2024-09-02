#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm
import sys
import yaml
from gc import collect

from train import load_data
import models as M


# %%
def get_hessian(model, x, x_prime, l, k, p, m):
    # TODO input to f(x) in formula
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (1,-1))

    with tf.GradientTape() as t2:
        t2.watch(x)
        with tf.GradientTape() as t1:
            t1.watch(x)
            y_pred = model(x_prime + ((l/k) * (p/m) * (x - x_prime)))
            class_score = tf.reduce_max(y_pred)
            grad = t1.gradient(class_score, x)
        hessian = t2.jacobian(grad, x)
        hessian_np = hessian.numpy()
        #print('hessian_np shape:', hessian_np.shape)
    return hessian_np

# %%
def integrated_hessians(model, x, x_prime):
    # TODO for one sample

    # number of points to approximate integral
    k = len(x)
    # m samples
    m = len(x)

    for i in range(m):
        for j in range(m):
            integrated_sum = 0
            for l in range(k):
                alphas = 0
                beta = 0
                for p in range(m):
                    # no integer div
                    l_ = np.float32(l)
                    m_ = np.float32(m)
                    k_ = np.float32(k)
                    p_ = np.float32(p)

                    alpha = (l/k) * (p/m) * get_hessian(model, x, x_prime, l_, k_, p_, m_) * (1/(k_*m_))
                    alphas += alpha
                beta += alphas
            integrated_sum += beta

            interaction_value = (x[i] - x_prime[i]) * (x[j] - x_prime[j]) * integrated_sum
            print('computed interaction value for sample', i, ':', interaction_value)
    return interaction_value

# %%
if __name__ == '__main__':
    start_time = datetime.now()

    DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-03/49101680_outbound_dnn_run_2023-10-03 13:37:10.673149/checkpoint100-0.06.h5'
    DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-03/49101680_outbound_dnn_run_2023-10-03 13:37:10.673149/params.yml'

    # load dnn model params
    with open(DNN_PARAMS_PATH) as file:
        params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # init dnn model with params
    model = M.get_dnn_model(params)

    # load dnn model weights
    model.load_weights(DNN_MODEL_PATH)

    print(model.summary())

    # load data
    DIRECTION = 'outbound' 
    _, _, _, _, x_test, y_test = load_data(DIRECTION)


    #%%
    # FIXME use first sample as baseline x'
    x_prime = x_test[0]

    interactions = []
    for i in tqdm(range(len(x_test))):
        interaction = integrated_hessians(model, x_test[i], x_prime)
        interactions.append(interaction)
        del interaction
        collect()
    interactions_np = np.array(interactions)
    del interactions
    collect()

    #%%
    print('interactions shape:', interactions_np.shape)
    sys.stdout.flush()
    mean_interactions = np.mean(interactions_np, axis=1)

    print('mean interactions shape:', mean_interactions.shape)
    print(mean_interactions)

    np.save('mean_interactions.npy', mean_interactions)
    np.save('interactions.npy', interactions_np)
    
    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)
# %%
