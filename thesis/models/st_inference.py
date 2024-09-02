import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from sklearn.metrics import classification_report
from joblib import load
from datetime import datetime, date
import os

DATA_DIR = '/data/alexander.huang/data/0903_data/'
TRAIN_DATA_DIR = DATA_DIR+'train/'
VAL_DATA_DIR = DATA_DIR+'val/'
TEST_DATA_DIR = DATA_DIR+'test/'

def weighted_ce_loss(y_true, y_pred):
    cce = tf.losses.CategoricalCrossentropy()
    soft_weight = 1
    # use lower weight -> best results per Hinton paper
    hard_weight = 1 
    print('soft weight:', soft_weight, '- hard weight', hard_weight)
    # format of true labels: 0,1 is soft label, 2,3 is hard label
    # e.g. [[.32, .68, 0, 1]]
    y_true_soft = y_true[:,0:2] 
    y_true_hard = y_true[:,2:4] 

    soft_loss = cce(y_true_soft, y_pred)
    hard_loss = cce(y_true_hard, y_pred)

    # weighted sum of cross-entropy loss of the soft labels and the hard labels
    total_loss = soft_weight * soft_loss + hard_weight * hard_loss

    return total_loss

if __name__ == '__main__':
    start = datetime.now()
    # take model from input
    direction = 'outbound'
    model_name = 'stdnn' 

    JOB_ID = sys.argv[1]

    x_train_path = TRAIN_DATA_DIR + direction + '_x_train.parq' 
    x_val_path = VAL_DATA_DIR + direction + '_x_val.parq' 

    x_path = TEST_DATA_DIR + direction + '_x_test.parq' 
    y_path = TEST_DATA_DIR + direction + '_y_test.parq' 
    print('x path:', x_path)
    print('y path:', y_path)
    sys.stdout.flush()

    # load data
    print('loading x data...')
    sys.stdout.flush()
    x_test = pd.read_parquet(x_path)
    x_test = x_test.to_numpy()
    x_train = pd.read_parquet(x_train_path)
    x_val = pd.read_parquet(x_val_path)

    print('loading y test data...')
    sys.stdout.flush()
    y_test = pd.read_parquet(y_path)
    y_test = y_test.to_numpy()

    # load model
    print('loading model...')
    sys.stdout.flush()
    #model = load(model_name+'.joblib')
    # .8 f1 best model 9/7
    model_path='/smallwork/alexander.huang/models/2023-09-07/48967256_outbound_stdnn_run_2023-09-07 00:24:16.720624/checkpoint198-0.25.h5'
    loaded_model = tf.keras.models.load_model(model_path, safe_mode=False, custom_objects={'weighted_ce_loss': weighted_ce_loss})
    loaded_model.summary()

    model = tf.keras.models.Sequential()
    layers = [l for l in loaded_model.layers]
    l1 = layers[:5]
    l2 = [layers[6]]
    layers = l1 + l2
    for layer in layers:
        model.add(layer)
    model = tf.keras.Model(inputs=model.input, outputs=model.output)
    model.summary()

    #y_pred = model.predict_proba(x_test)
    y_pred_proba = model.predict(x_test, verbose=2)

    # for softmax prediction
    y_pred_classes = tf.argmax(y_pred_proba, axis=1)
    y_pred_classes = y_pred_classes.numpy()

    print('Metrics:')
    print(classification_report(y_test, y_pred_classes, digits=4))
    sys.stdout.flush()

    results = pd.DataFrame(data=y_pred_proba, columns=['label_0', 'label_1'])
    sys.stdout.flush()

    save_dir = 'student_teacher_results/' + JOB_ID + '/' 
    os.makedirs(save_dir, exist_ok=True)

    out_path = save_dir + model_name + '_predictions.parq'
    results.to_parquet(out_path)
    sys.stdout.flush()

    y_train_pred = model.predict(x_train, verbose=2)
    y_train_out = pd.DataFrame(data=y_train_pred, columns=['model_train_pred_0', 'model_train_pred_1'])
    y_train_out.to_parquet(save_dir+'student_learned_training_labels.parq')

    y_val_pred = model.predict(x_val, verbose=2)
    y_val_out = pd.DataFrame(data=y_val_pred, columns=['model_val_pred_0', 'model_val_pred_1'])
    y_val_out.to_parquet(save_dir+'student_learned_val_labels.parq')

    y_pred_df = pd.DataFrame(data=y_pred_proba, columns=['model_test_pred_0', 'model_test_pred_1'])
    y_pred_df.to_parquet(save_dir+'student_learned_test_labels.parq')
    

    end = datetime.now()
    total = end - start
    print('Start time:', start)
    print('End time:', end)
    print('Total time:', total)