import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import yaml
import sys
import pandas as pd
import os
from tqdm import tqdm

import models as M
from train import load_data


if __name__ == '__main__':
    # load data
    # first argument is direction: inbound or outbound
    DIRECTION = sys.argv[1] 
    # second argument is dropout model: mcd or concrete
    MODEL = sys.argv[2]
    # third argument is job id
    JOB_ID = sys.argv[3] 


    if DIRECTION == 'outbound':
        if MODEL == 'concrete':
            SAVE_DIR = '/smallwork/alexander.huang/concrete_dropout/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained concrete dropout model
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595398_outbound_cdnn_run_2023-11-18 14:25:18.556080/checkpoint100-0.06.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595398_outbound_cdnn_run_2023-11-18 14:25:18.556080/params.yml'
        elif MODEL == 'mcd':
            # NOTE: used mcd code for published runs in train.py instead
            SAVE_DIR = '/smallwork/alexander.huang/mcdropout/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained dnn model with standard dropout
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-13/49111756_outbound_dnn_run_2023-10-13 13:27:28.481509/checkpoint100-0.06.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-13/49111756_outbound_dnn_run_2023-10-13 13:27:28.481509/params.yml'
        elif MODEL == 'vi':
            SAVE_DIR = '/smallwork/alexander.huang/vi/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained vi model
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-23/49132897_outbound_vidnn_run_2023-10-23 17:20:40.887884/checkpoint400-0.05.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-23/49132897_outbound_vidnn_run_2023-10-23 17:20:40.887884/params.yml'

    elif DIRECTION == 'inbound':
        if MODEL == 'concrete':
            SAVE_DIR = '/smallwork/alexander.huang/concrete_dropout/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained concrete dropout model
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595399_inbound_cdnn_run_2023-11-18 14:25:32.801963/checkpoint287-0.14.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595399_inbound_cdnn_run_2023-11-18 14:25:32.801963/params.yml'
        elif MODEL == 'mcd':
            SAVE_DIR = '/smallwork/alexander.huang/mcdropout/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained dnn model with standard dropout
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-16/49112464_inbound_dnn_run_2023-10-16 12:58:36.901380/checkpoint100-0.14.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-16/49112464_inbound_dnn_run_2023-10-16 12:58:36.901380/params.yml'
        elif MODEL == 'vi':
            SAVE_DIR = '/smallwork/alexander.huang/vi/'
            RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
            os.makedirs(RUN_DIR, exist_ok=True)
            # trained vi model
            MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-23/49119487_inbound_vidnn_run_2023-10-23 13:31:14.115731/checkpoint798-0.06.h5'
            PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-23/49119487_inbound_vidnn_run_2023-10-23 13:31:14.115731/params.yml'

    # load test data
    _, _, _, _, x_test, y_test = load_data(DIRECTION)


    # load model config
    with open(PARAMS_PATH) as file:
        params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # load model and saved wightss
    if MODEL == 'concrete':
        model = M.get_cdnn_model(params)
        print('MC inference w/ Concrete Dropout')
    elif MODEL == 'mcd':
        model = M.get_dnn_model(params)
        print('MC inference w/ MC Dropout')
    elif MODEL == 'vi':
        model = M.get_vi_model(params)
        print('MC inference w/ vi model')

    model.summary()
    model.load_weights(MODEL_PATH)

    n_models = 20
    print('Testing with', n_models, 'models...')
    sys.stdout.flush()

    if MODEL == 'vi':
        y_probas = np.stack([model.predict(x_test, verbose=2) for _ in tqdm(range(n_models))])
    elif MODEL == 'concrete':
        y_probas = np.stack([model.predict(x_test, verbose=2) for _ in tqdm(range(n_models))])
    elif MODEL == 'mcd':
        y_probas = np.stack([model(x_test, training=True) for _ in tqdm(range(n_models))])

    print('Probs shape:', y_probas.shape)
    y_proba = y_probas.mean(axis=0)
    print('Probs mean shape:', y_proba.shape)
    y_pred_classes = np.argmax(y_proba, axis=1)
    print('Class predictions shape:', y_pred_classes.shape)

    print('Aggregate test predictions:\n', y_pred_classes)
    sys.stdout.flush

    if MODEL == 'concrete':
        ps = np.array([keras.backend.eval(layer.p_logit) for layer in model.layers if hasattr(layer, 'p_logit')])
        droput_val = tf.nn.sigmoid(ps).numpy()
        print('Learned Dropout Values:', droput_val)
        print('\nTest Data Results for concrete dropout model:')
    elif MODEL == 'mcd':
        print('\nTest Data Results for MC dropout model:')
    elif MODEL == 'vi':
        print('\nTest Data Results for vi model:')

    rep = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_classes, digits=4))
    report.to_csv(RUN_DIR + 'classification_report.csv')
    sys.stdout.flush()

    print('saving results')
    np.save(RUN_DIR + 'predictions.npy', y_probas)
    print('done saving.')
