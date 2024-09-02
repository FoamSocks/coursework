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
import os

from train import load_data
import models as M

#%%
def get_gradient(model, x):
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (1,-1))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        # get score for this class
        # class_score = y_pred[0][S_c] 
        class_score = tf.reduce_max(y_pred)
        #print('class score: ', class_score.numpy())
        # calculate dS/dx
        gradient = tape.gradient(class_score, x)
    return gradient

#%%
def get_saliency_map(model, x):
    saliencymap = [] 
    if sys.argv[2] == 'all':
        num_maps = int(len(x))
    else:
        num_maps = int(sys.argv[2]) 
    for i in tqdm(range(num_maps)):
        dS_dx = get_gradient(model, x[i])
        grad = dS_dx.numpy().flatten()
        saliencymap.append(grad)
    saliencymap = np.array(saliencymap)
    return saliencymap


#%%
if __name__ == '__main__':
    start_time = datetime.now()

    SAVE_DIR = '/smallwork/alexander.huang/saliency_maps/'
    DIRECTION = sys.argv[1] 
    if DIRECTION == 'outbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/checkpoint97-0.06.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/params.yml'
    elif DIRECTION == 'inbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/checkpoint82-0.14.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/params.yml'
    else:
        raise ValueError('Invalid dataset, received:', DIRECTION)

    NUM_SAMPLES = sys.argv[2]
    JOB_ID = sys.argv[3]
    RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + NUM_SAMPLES + '/' + JOB_ID + '/'
    os.makedirs(RUN_DIR, exist_ok=True)

    # load dnn model params
    with open(DNN_PARAMS_PATH, 'r') as file:
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
    # generate saliency maps for respective classes, 0 and 1
    allow_saliency_map = get_saliency_map(model, x_test_allow)
    block_saliency_map = get_saliency_map(model, x_test_block)
    
    #%%
    print('allow shape:', allow_saliency_map.shape)
    sys.stdout.flush()
    mean_allow = np.mean(allow_saliency_map, axis=0)

    print('mean allow shape:', mean_allow.shape)
    print(mean_allow)

    np.save(RUN_DIR + 'allow_mean_saliency.npy', mean_allow)
    np.save(RUN_DIR + 'allow_saliency_maps.npy', allow_saliency_map)

    print('block shape:', block_saliency_map.shape)
    sys.stdout.flush()
    mean_block = np.mean(block_saliency_map, axis=0)

    print('mean allow shape:', mean_block.shape)
    print(mean_block)

    np.save(RUN_DIR + 'block_mean_saliency.npy', mean_block)
    np.save(RUN_DIR + 'block_saliency_maps.npy', block_saliency_map)
    
    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)
# %%
