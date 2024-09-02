#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import sys
import os
import yaml
from gc import collect

from train import load_data
import models as M


# %%
def get_hessian(model, x):
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (1,-1))

    with tf.GradientTape() as t2:
        t2.watch(x)
        with tf.GradientTape() as t1:
            t1.watch(x)
            y_pred = model(x)
            #$class_score = y_pred[0][S_c]
            class_score = tf.reduce_max(y_pred) 
            grad = t1.gradient(class_score, x)
        hessian = t2.jacobian(grad, x)
        #print('hessian_np shape:', hessian_np.shape)
        # return tf hessian
    return hessian

# %%

def get_interactions(model, x):
    interactions = []
    if sys.argv[2] == 'all':
        rng = len(x)
    else:
        np.random.shuffle(x)
        rng = int(sys.argv[2])

    for i in tqdm(range(rng)):
        interaction = get_hessian(model, x[i])
        # convert tf to numpy
        interaction_np = interaction.numpy()
        interactions.append(interaction_np)
        del interaction
        del interaction_np
        collect()
    interactions_np = np.array(interactions).squeeze()
    del interactions
    collect()
    return interactions_np

#%%
if __name__ == '__main__':
    start_time = datetime.now()

    SAVE_DIR = '/smallwork/alexander.huang/vanilla_hessians/'
    DIRECTION = sys.argv[1] 
    if DIRECTION == 'outbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/checkpoint97-0.06.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/params.yml'
    elif DIRECTION == 'inbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/checkpoint82-0.14.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/params.yml'
    else:
        raise ValueError('Invalid dataset, recieved:', DIRECTION)

    JOB_ID = sys.argv[3]
    RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + sys.argv[2] + '/' + sys.argv[3] + '/' 
    os.makedirs(RUN_DIR, exist_ok=True)

    # load dnn model params
    with open(DNN_PARAMS_PATH, 'rb') as file:
        params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # init dnn model with params
    model = M.get_dnn_model(params)

    # load dnn model weights
    model.load_weights(DNN_MODEL_PATH)

    print(model.summary())

    # load data
    _, _, _, _, x_test, y_test = load_data(DIRECTION)

    y_pred = model.predict(x_test, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)

    #%%
    blocked_index = np.where(y_pred_classes == 1)
    x_test_allow = np.delete(x_test, blocked_index, axis=0)
    print('allowed:', x_test_allow.shape)
    x_test_block = np.take(x_test, blocked_index, axis=0)
    x_test_block = np.squeeze(x_test_block)
    print('blocked:', x_test_block.shape)

    #%%
    interactions_allow = get_interactions(model, x_test_allow)
    interactions_block = get_interactions(model, x_test_block)

    #%%
    print('class 0 interactions shape:', interactions_allow.shape)
    sys.stdout.flush()
    mean_allow_interactions = np.mean(interactions_allow, axis=0)
    print('mean allow interactions shape:', mean_allow_interactions.shape)

    np.save(RUN_DIR + 'allow_mean_vanilla_interactions.npy', mean_allow_interactions)
    np.save(RUN_DIR + 'allow_vanilla_interactions.npy', interactions_allow)

    print('class 1 interactions shape:', interactions_block.shape)
    sys.stdout.flush()
    mean_block_interactions = np.mean(interactions_block, axis=0)
    print('mean block interactions shape:', mean_block_interactions.shape)

    np.save(RUN_DIR + 'block_mean_vanilla_interactions.npy', mean_block_interactions)
    np.save(RUN_DIR + 'block_vanilla_interactions.npy', interactions_block)

    ''' visualization code
    def imshow_zero_center(hessian, **kwargs):
        lim = tf.reduce_max(abs(hessian))
        plt.imshow(hessian, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
        plt.colorbar()

    # remove interactions < 1e-4
    threshold = 1e-2
    indices = np.where(np.abs(mean_interactions[0]) < threshold)
    print(indices)
    filter_interactions = np.delete(mean_interactions, indices, axis=0)
    filter_interactions = np.delete(filter_interactions, indices, axis=1)
    imshow_zero_center(filter_interactions)
    '''

    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)
# %%
