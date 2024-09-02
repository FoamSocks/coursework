'''
student_teacher.py

Implements student-teacher method of knowledge distillation per https://arxiv.org/abs/1503.02531 (G. Hinton).
Loads a random forest classifier as the teacher, uses test data to generate labels from RFC predictions.
Then trains a DNN with soft labels from RFC predictions weighted with hard true labels.

Two major findings that made this work:
    1. Custom weighted cross-entropy function was used to weight the sum of cross-entropy loss
       between the soft labels (rfc predictions) and hard labels (true label).
    2. High softmax temperature (8) was used. Temperature adjustment was implemented with 
       a Keras Lambda layer.

st_inference.py removes the softmax temperature adjustment per Hinton paper by removing the Lambda layer.

See model.py for model implementation and weighted cross-entropy loss function.

Training code was separated from train.py due to this code being used as part of a course project for
CS4321.

This procedure was tested and working with soft/hard label weights at 1, 1, 
but was deemed unnecessary as DNN began to outperform RFC classifiers
once the right train/test/validation split was determined.
'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
import json
from joblib import load
from datetime import datetime, date
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from joblib import dump, load
from collections import Counter

import models as M
import callbacks
from train import load_data, sampling

if __name__ == '__main__':
    start = datetime.now()
    JOB_ID = sys.argv[1]

    # take model from input
    direction = 'outbound' 
    model_name = '/home/alexander.huang/smallwork/models/2023-09-04/48957770_outbound_rfc_run_2023-09-04 21:04:53.112374/outbound_rfc_.joblib'

    # dnn
    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_stdnn_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)

    print('Parameters:')
    params = {}
    if direction == 'outbound':
        # parameters
        reg_strength = 1e-4
        params = {'n_neurons':512,
                  'n_drop':0.2,
                  'activation':'swish',
                  'eta':1e-6,
                  'regularizer':keras.regularizers.L2(reg_strength),
                  'b_size': 4096,
                  'n_epochs': 200, 
                  'sample_ratio': 0.1,
                  'sample_method':'smote'
                  }

        params['reg_strength'] = reg_strength 
        print(params)

    elif direction == 'inbound':
        reg_strength = 1e-4
        params = {'n_neurons':512,
                  'n_drop':0.2,
                  'activation':'swish',
                  'eta':1e-6,
                  'regularizer':keras.regularizers.L2(reg_strength),
                  'b_size':4096,
                  'n_epochs':100,
                  'sample_ratio':1.0,
                  'sample_method':'ros'
                  }
        params['reg_strength'] = reg_strength 
        print(params)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(direction)
    x_train, y_train = sampling(x_train, y_train, params)

    # load saved model
    print('loading model...')
    print('model: ', model_name)
    sys.stdout.flush()
    model = load(model_name)
    #model = tf.keras.models.load_model('my_model.keras')

    # predict, and grab one-hot encoded probabilities
    y_pred = model.predict_proba(x_train)
    y_train_ohe = tf.one_hot(y_train, 2).numpy()
    y_train_stacked = np.hstack((y_pred, y_train_ohe))
    
    y_pred_val = model.predict_proba(x_val)
    y_val_ohe = tf.one_hot(y_val, 2).numpy()
    y_val_stacked = np.hstack((y_pred_val, y_val_ohe))

    y_pred_test = model.predict_proba(x_test)

    # RFC predictions on training data
    results = pd.DataFrame(data=y_pred, columns=['label_0', 'label_1'])
    #y_train_soft = results['label_1'].to_frame()
    y_train_soft = results
    y_train_soft.to_parquet(save_dir + 'teacher_training_labels.parq')

    # RFC predictions on validation data
    results = pd.DataFrame(data=y_pred_val, columns=['label_0', 'label_1'])
    #y_val_soft = results['label_1'].to_frame()
    y_val_soft = results
    y_val_soft.to_parquet(save_dir + 'teacher_validation_labels.parq')

    # RFC predictions on test data
    results = pd.DataFrame(data=y_pred_test, columns=['label_0', 'label_1'])
    #y_test_soft = results['label_1'].to_frame()
    y_test_soft = results
    y_test_soft.to_parquet(save_dir + 'teacher_test_labels.parq')

    x_train = tf.cast(x_train, tf.float32)
    #y_train = tf.cast(y_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)
    #y_val = tf.cast(y_val, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    #y_test = tf.cast(y_test, tf.float32)
    y_train_soft = tf.cast(y_train_soft.to_numpy(), tf.float32)
    y_val_soft = tf.cast(y_val_soft.to_numpy(), tf.float32)
    y_test_soft = tf.cast(y_test_soft.to_numpy(), tf.float32)

    input_size = x_train.shape[1]
    params['input_size'] = input_size
    print('\nInput Size:', input_size)
    sys.stdout.flush()
    # FIXME load saved model
    #saved_model = '/home/alexander.huang/repositories/huang-thesis/models/48950875_outbound_stdnn_run_2023-08-28 10:49:53.851112/outbound_stdnn_model.keras'
    #model = keras.models.load_model(saved_model)

    model = M.get_stdnn_model(params)

    print(model.summary())
    sys.stdout.flush()

    print('----- '+direction+' traffic deterministic neural network training -----')
    sys.stdout.flush()

    n_epochs = params['n_epochs']
    print('Total number of epochs:', n_epochs)
    sys.stdout.flush()
    history = model.fit(x_train, y_train_stacked,
                        epochs=n_epochs,
                        validation_data=(x_val, y_val_stacked),
                        batch_size=params['b_size'],
                        verbose=2,
                        callbacks=callbacks.make_callbacks(save_dir))
    
    print('Saving '+direction+' st-dnn model...')

    # write params to params.txt
    params['regularizer'] = str(params['regularizer'])
    with open(save_dir+'/params.json', 'w') as file:
        file.write(json.dumps(params))
    
    # save model
    model_path = save_dir + '/' + direction +'_stdnn_model.keras'
    model.save(model_path)
    print('Save complete.')
    print('Model path:', model_path)
    sys.stdout.flush()    
    # save metrics history
    print('Saving model history....')
    dump(history, save_dir+'/'+direction +'_stdnn_history')

    # plot training curves
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title(direction + ' dnn loss')
    plt.ylabel('crossentropy loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.savefig(save_dir+'/'+direction+'_dnn_loss.png', bbox_inches='tight')

    print('================== ' + direction + ' model results ==================')
    y_train_pred = model.predict(x_train, verbose=2)
    y_train_out = pd.DataFrame(data=y_train_pred, columns=['model_train_pred_0', 'model_train_pred_1'])
    y_train_out.to_parquet(save_dir+'student_learned_training_labels.parq')

    y_val_pred = model.predict(x_val, verbose=2)
    y_val_out = pd.DataFrame(data=y_val_pred, columns=['model_val_pred_0', 'model_val_pred_1'])
    y_val_out.to_parquet(save_dir+'student_learned_val_labels.parq')

    y_pred = model.predict(x_test, verbose=2)
    y_pred_df = pd.DataFrame(data=y_pred, columns=['model_test_pred_0', 'model_test_pred_1'])
    y_pred_df.to_parquet(save_dir+'student_learned_test_labels.parq')

    # flatten to process into single binary label for classification report
    #y_pred_proba = y_pred.flatten()
    # convert sigmoid probabilities into classes with thresh = 0.5
    #y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)

    # convert softmax probabilities to single label
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_pred_classes = y_pred_classes.numpy()
    print('Test Data Results:')
    print('y_test', y_test)
    print('true classes:')
    print(sorted(Counter(y_test).items()))

    print('y_pred', y_pred_classes)
    print('pred classes:')
    print(sorted(Counter(y_pred_classes).items()))

    rep = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_classes, digits=4))
    report.to_csv(save_dir+'/'+direction+'_stdnn_report.csv')
    sys.stdout.flush()

    end = datetime.now()
    total = end - start
    print('Start time:', start)
    print('End time:', end)
    print('Total time:', total)
