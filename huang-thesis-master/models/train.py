'''
train.py

Loads proccessed training, validation, and test data from file and performs model
training with various models.

Usage:
    train.py <direction> <model> <slurm_job_id>

Pass any integer for slurm_job_id if not using slurm batch job.
Otherwise batch job should pull job identifier from bash variable $SLURM_JOB_ID.

direction:  inbound OR outbound train/test/validation datasets will be loaded

model:      rfc             sklearn random forest classifier
            gbrfc           sklearn gradient boosted RFC (unused)
            trfc            Keras random forest classifier (unused)
            tuner_dnn       DNN with Keras Tuner
            dnn             DNN with standard dropout
            mcd             DNN with Monte Carlo dropout
            nn_vi           DNN with Gaussian variational inference (unused)
            cdnn            DNN with concrete dropout

Adjust model parameters within each function.

Model parameters are saved as .yml for rfc, dnn, mcd, nn_vi, and cdnn models.
Other models were dropped from the thesis paper and have not been updated since.
The functions may not be compatible with current repository or current versions of 
TensorFlow, Keras, or sklearn.

For TensorFlow/Keras models, model weights will be saved as .h5.
For sklearn models, seralized model is saved as .joblib.
'''
import numpy as np
import sklearn as skl
import sys
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner
from datetime import datetime, date
from collections import Counter
from sklearn.metrics import classification_report
from joblib import dump, load

import yaml

from gc import collect

# custom functions
import models as M
import callbacks

# disable tf logging for info messages
import os
print('tf INFO messages are suppressed.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#print('CUDA devices:')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras

print('sklearn version:', skl.__version__)
sys.stdout.flush()

start_time = datetime.now()

# directory of processed data from feature engineering
DATA_DIR = '/data/alexander.huang/data/0903_data/'

def load_data(direction: str) -> tuple:
    '''
    Loads training, validation, and test datasets from parquet file determined in DATA_DIR.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

    Returns:
        x_train (array-like): Loaded training data features.

        y_train (array-like): Loaded training data labels.

        x_val (array-like): Loaded validation data features.

        y_val (array-like): Loaded validation data labels.

        x_test (array-like): Loaded test data features.

        y_test (array-like): Loaded test data labels.

    Raises:
        ValueError: If direction is not "inbound" or "outbound."
    '''

    if not (direction == 'inbound' or direction == 'outbound'):
        raise ValueError(f'invalid direction, recieved: {direction}')

    # load training, validation, and test data
    # mod: without timestamp and CC_NOT_IN_DB feature, and TTL
    X_VAL_FILE = DATA_DIR + 'val_drop/' + direction + '_x_val.parq'
    Y_VAL_FILE = DATA_DIR + 'val_drop/' + direction + '_y_val.parq'
    X_TRAIN_FILE = DATA_DIR + 'train_drop/' + direction + '_x_train.parq'
    Y_TRAIN_FILE = DATA_DIR + 'train_drop/' + direction + '_y_train.parq'
    X_TEST_FILE = DATA_DIR + 'test_drop/' + direction + '_x_test.parq'
    Y_TEST_FILE = DATA_DIR + 'test_drop/' + direction + '_y_test.parq'

    print('\n======== loadiing train and test data ==========')
    sys.stdout.flush()

    print('load x_train')
    sys.stdout.flush()
    x_train = pd.read_parquet(X_TRAIN_FILE)
    print('load y_train')
    sys.stdout.flush()
    y_train = pd.read_parquet(Y_TRAIN_FILE)
    sys.stdout.flush()

    print('load x_test')
    x_test = pd.read_parquet(X_TEST_FILE)
    sys.stdout.flush()
    print('load x_test')
    y_test = pd.read_parquet(Y_TEST_FILE)

    print('load x_val')
    x_val = pd.read_parquet(X_VAL_FILE)
    print('load y_val')
    y_val = pd.read_parquet(Y_VAL_FILE)

    print('...load complete.')
    sys.stdout.flush()

    # convert dataframes to numpy
    x_train = x_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32).reshape(1,-1).ravel()

    x_test = x_test.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32).reshape(1,-1).ravel()

    x_val = x_val.to_numpy(dtype=np.float32)
    y_val = y_val.to_numpy(dtype=np.float32).reshape(1,-1).ravel()
    
    print('imported training data:')
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(sorted(Counter(y_train).items()))

    print('imported validation data:')
    print('x_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape)
    print(sorted(Counter(y_val).items()))

    print('imported test data:')
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print(sorted(Counter(y_test).items()))
    sys.stdout.flush()

    return x_train, y_train, x_val, y_val, x_test, y_test 

def oversample(x: any, y: any, sample_ratio: float):
    '''
    Performs random over-sampling given input x and y numpy arrays to some sample ratio.
    Unused in final thesis paper.

    Args:
        x (array-like): Array-like of features

        y (array-like): Array-like of labels

        sample_ratio (float): Ratio to over-sample the under-represented class

    Returns:

        x_oversampled (array-like): Over-sampled array-like of features

        y_oversampled (array-like): Over-sampled array-like of labels
    '''
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=None, sampling_strategy=sample_ratio)
    x_oversampled, y_oversampled = ros.fit_resample(x,y)
    del x
    del y
    collect()
    print('sample ratio:', sample_ratio)
    print('\nCounts for random oversampling:')
    print(sorted(Counter(y_oversampled).items()))
    sys.stdout.flush()
    return x_oversampled, y_oversampled

def undersample(x: any, y: any, sample_ratio: float):
    '''
    Performs random under-sampling given input x and y numpy arrays to some sample ratio.
    Unused in final thesis paper.

    Args:
        x (array-like): Array-like of features

        y (array-like): Array-like of labels

        sample_ratio (float): Ratio to under-sample the over-represented class

    Returns:

        x_undersampled (array-like): Under-sampled array-like of features

        y_undersampled (array-like): Under-sampled array-like of labels
    '''

    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=None, sampling_strategy=sample_ratio)
    x_undersampled, y_undersampled = rus.fit_resample(x,y)
    print('\nCounts for random undersampling:')
    print(sorted(Counter(y_undersampled).items()))
    sys.stdout.flush()
    return x_undersampled, y_undersampled

def smote(x, y, sample_ratio):
    '''
    Performs SMOTE over-sampling given input x and y numpy arrays to some sample ratio.
    SMOTE: https://arxiv.org/abs/1106.1813

    Args:
        x (array-like): Array-like of features

        y (array-like): Array-like of labels

        sample_ratio: Ratio to over-sample the under-represented class

    Returns:

        x_oversampled (array-like): Over-sampled array-like of features

        y_oversampled (array-like): Over-sampled array-like of labels
    '''

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=None, sampling_strategy=sample_ratio)
    x_oversampled, y_oversampled = smote.fit_resample(x,y)
    print('sample ratio:', sample_ratio)
    print('\nCounts for SMOTE oversampling:')
    print(sorted(Counter(y_oversampled).items()))
    sys.stdout.flush()
    return x_oversampled, y_oversampled

def sampling(x_train: any, y_train: any, params: dict):
    '''
    Handler function for sampling methods. Supports random over-sampling,
    random under-sampling, and SMOTE over-sampling. Also supports
    bypassing sampling.

    Args:
        x_train (array-like): Training features to sample.
        
        y_train (array-like): Training labels to sample

        params (dict): Dictionary of parameters loaded from YAML for model training.

    Returns:
        x_train (array-like): Sampled training features.

        y_train (array-like): Sampling training labels.

    Raises:
        ValueError: If sample method provided in loaded params file is not
        "ros", "rus", "smote", or "no_sample".
    '''
    ratio=params['sample_ratio']
    if params['sample_method'] == 'ros':
        # ROS
        print('=== Random Oversampling ===')
        x_train, y_train = oversample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'rus':
        # undersampling is trash
        # RUS
        print('=== Random Undersampling ===')
        x_train, y_train = undersample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'smote':
        print('=== SMOTE Oversampling ===')
        x_train, y_train = smote(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'no_sample':
        print('=== no sampling method selected ===')
        sys.stdout.flush()
    else:
        raise ValueError(f'invalid sampling method, received: {params["sample_method"]}')
    return x_train, y_train

''' unused logistic regression model code, kept for reference.
def log_reg_model(direction, x_train, y_train, x_val, y_val, x_test, y_test):
    from sklearn.linear_model import SGDClassifier 

    # logistic regression with L2 regularization
    lg_model = SGDClassifier(loss='log_loss', 
                            eta0=1e-7, 
                            learning_rate='optimal', 
                            warm_start=True, 
                            penalty='l2')

    print('----- ' + direction + ' traffic logistic regression -----')
    sys.stdout.flush()

    n_epochs = 10 
    print('Total number of epochs:', n_epochs)
    sys.stdout.flush()
    max_accuracy = 0
    acc_plot_train_values = []
    acc_plot_val_values = []
    best_epoch = None
    best_model = None
    for epoch in range(n_epochs):
        print(direction, 'logistic regression, epoch:', epoch)
        sys.stdout.flush()
        # fit model on training data
        lg_model.partial_fit(x_train, y_train, classes=np.unique(y_train))
        # make prediction
        y_pred_train = lg_model.predict(x_train)
        y_pred_val = lg_model.predict(x_val)
        # calculate training accuracy and save for plot
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        recall_train = recall_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
        print('training:: accuracy:', accuracy_train, 'precision:', precision_train, 'recall:', recall_train, 'f1 score:', f1_train)
        sys.stdout.flush()
        acc_plot_train_values.append(accuracy_train)
        # calculate validation accuracy and save for plot
        accuracy_val = accuracy_score(y_val, y_pred_val)
        precision_val = precision_score(y_val, y_pred_val)
        recall_val = recall_score(y_val, y_pred_val)
        f1_val = f1_score(y_val, y_pred_val)
        print('validation:: accuracy:', accuracy_val, 'precision:', precision_val, 'recall:', recall_val, 'f1 score:', f1_val)
        sys.stdout.flush()
        acc_plot_val_values.append(accuracy_val)
        # compare results of epoch to overall accuracy, save best accuracy model
        if max_accuracy < accuracy_val:
            max_accuracy = accuracy_val
            best_epoch = epoch
            best_model = deepcopy(lg_model)

    print('======================= results ====================')
    print('Best Epoch:', best_epoch)
    print('Max Val Accuracy:', max_accuracy)

    # plot learning results
    x = np.array(range(n_epochs))
    plt.xlim(0,n_epochs)
    plt.ylim(0,1)
    plt.plot(x, acc_plot_val_values, label='Validation', marker='o', markevery=[best_epoch])
    #plt.text(50,5000,"Best Model")
    plt.plot(x, acc_plot_train_values, label='Train')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.margins(0.3)
    if direction == 'inbound':
        plt.savefig('inbound_log_reg_training.png', bbox_inches='tight')
    elif direction == 'outbound':
        plt.savefig('outbound_log_reg_training.png', bbox_inches='tight')

    # test best model
    y_pred_test = best_model.predict(x_test)

    print('================== best model results on test set ==================')
    print(classification_report(y_test, y_pred_test, digits=5))
    sys.stdout.flush()    

    print('Saving best model...')
    # https://scikit-learn.org/stable/model_persistence.html
    sys.stdout.flush()
    if direction == 'inbound':
        dump(best_model, 'inbound_logistic_regression_model.joblib')
    elif direction == 'outbound':
        dump(best_model, 'outbound_logistic_regression_model.joblib')

    print('Save complete.')
    sys.stdout.flush()

    return


def log_vi_model(data_option, x_train, y_train, x_val, y_val, x_test, y_test):
    import tensorflow_probability as tfp
    from tensorflow import keras

    params = {}
    if direction == 'outbound':
        # parameters
        reg_strength = 1e-5
        params = {'n_neurons':256,
                  'n_drop':0.4,
                  'activation':'swish',
                  'eta':1e-7,
                  'regularizer':keras.regularizers.L2(reg_strength),
                  'b_size':1024,
                  'n_epochs': 45, 
                  'sample_ratio': 1.0,
                  'sample_method':'ros'
                  }

        params['reg_strength'] = reg_strength 
        #print(params)
    elif direction == 'inbound':
        reg_strength = 0.01
        params = {'n_neurons':128,
                  'n_drop':0.5,
                  'activation':'swish',
                  'eta':1e-5,
                  'regularizer':keras.regularizers.L2(reg_strength),
                  'b_size':2048,
                  'n_epochs':100,
                  'sample_ratio':1.0,
                  'sample_method':'no_sample'
                  }

        params['reg_strength'] = reg_strength 
        #print(params)

    ratio=params['sample_ratio']
    if params['sample_method'] == 'ros':
        # ROS
        print('=== Random Oversampling ===')
        x_train, y_train = oversample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'rus':
        # undersampling is trash
        # RUS
        print('=== Random Undersampling ===')
        x_train, y_train = undersample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'no_sample':
        print('=== no sampling method selected ===')
        sys.stdout.flush()

    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)
    y_val = tf.cast(y_val, tf.float32)

    input_size = x_train.shape[1]
    params['input_size'] = input_size 
    print('input size:', input_size) 
    sys.stdout.flush()

    n_examples = y_train.shape[0]
    print('number of examples:', n_examples)
    sys.stdout.flush()
    expectNLL = lambda y, rv_y: -rv_y.log_prob(y)/n_examples

    def get_kernel_divergence_fn(train_size, kl_weight=1.0):
        def kernel_divergence_fn(q, p, _):
            kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
            return kl_weight * kernel_divergence
        return kernel_divergence_fn

    kl_div = get_kernel_divergence_fn(n_examples)

    model = M.get_vi_model(params, kl_div, expectNLL)

    print(model.summary())
    sys.stdout.flush()

    print('----- '+ direction +' traffic logistic regression VI training -----')
    sys.stdout.flush()

    print('Total number of epochs:', params['n_epochs'])
    sys.stdout.flush()
    history = model.fit(x_train, y_train,
                        epochs=params['n_epochs'],
                        validation_data=(x_val,y_val))

    print('Saving '+direction+' dnn model...')
    date = str(datetime.now())
    sys.stdout.flush()
    try:
        save_dir = direction+'_dnn_run_'+date
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    # write params to params.txt
    params['regularizer'] = str(params['regularizer'])
    with open(save_dir+'/params.json', 'w') as file:
        file.write(json.dumps(params))
    
    # save model
    model_path = save_dir + '/' + direction +'_dnn_model.keras'
    model.save(model_path)
    print('Save complete.')
    print('Model path:', model_path)
    sys.stdout.flush()    
    # save metrics history
    print('Saving model history....')
    dump(history, save_dir+'/'+direction +'_dnn_history')

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

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.grid(True)
    plt.title(direction + ' dnn metrics')
    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall'], loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.savefig(save_dir+'/'+direction+'_dnn_metrics.png', bbox_inches='tight')

    print('================== ' + direction + ' model results ==================')
    y_pred = model.predict(x_test, verbose=2)

    y_pred_proba = y_pred.flatten()
    # convert sigmoid probabilities into classes with thresh = 0.5
    y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)

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
    report.to_csv(save_dir+'/'+direction+'_dnn_report.csv')
    sys.stdout.flush()


    return
'''

def tf_rfc(direction: str, x_train: any, y_train: any, x_test: any, y_test: any):
    '''
    Training with TensorFlow/Keras random forest classifier instead of sklearn, 
    supports CUDA compute and Keras tuner in lieu of sklearn GridSearchCV 
    object.
    Unused in final thesis paper.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''
    import tensorflow_decision_forests as tfdf

    #tuner = tfdf.tuner.RandomSearch(num_trials=20)
    #tuner.discret('max_depth', [500, 600, 700])
    #model = tfdf.keras.RandomForestModel(tuner=tuner)
    print('====== tfdf '+direction+' RFC =====')
    sys.stdout.flush()
    if direction == 'outbound':
        params = {}
        params = {'num_trees':1000,
                  'max_depth':800,
                  'min_examples':10,
                  'min_split':10,
                  'sample_method':'smote',
                  'sample_ratio':1.0
        }
    elif direction == 'inbound':
        params = {'num_trees':100,
                  'max_depth':1000,
                  'min_examples':10,
                  'min_split':10,
                  'sample_method':'ros',
                  'sample_ratio':0.12 
        }

    ratio = params['sample_ratio']
    if params['sample_method'] == 'ros':
        # ROS
        print('=== Random Oversampling ===')
        x_train, y_train = oversample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'rus':
        # undersampling is trash
        # RUS
        print('=== Random Undersampling ===')
        x_train, y_train = undersample(x_train, y_train, ratio)
        sys.stdout.flush()
    elif params['sample_method'] == 'no_sample':
        print('=== no sampling method selected ===')
        sys.stdout.flush()

    model = tfdf.keras.RandomForestModel(
        num_trees=params['num_trees'],
        max_depth=params['max_depth'],
        categorical_set_split_max_num_items=params['min_split'],
        min_examples=params['min_examples']
    )

    model.compile()

    print('fitting model')
    sys.stdout.flush()
    history = model.fit(x_train, y_train, verbose=2)

    date = str(datetime.now())
    save_dir = direction + '_tfdf_rfc_run' + date

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass   

    print('Saving model history....')
    dump(history, save_dir+'/'+direction +'_dnn_history')

    print('dumping model')
    model.save(save_dir+'/'+direction+'_rfc_tfdf')

    '''
    with open(save_dir+'/params.json', 'w') as file:
        file.write(json.dumps(params))
    '''

    print('========== test results ==========')
    y_pred_proba = model.predict(x_test, verbose=2)
    y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)

    print('Test Data Results:')
    print('y_test')
    print(y_test)
    # count predictions
    print('true classes:')
    classes, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(classes, counts):
        print('class', label, ':', count)
    print('y_pred')
    print(y_pred_classes)
    print('pred classes:')
    classes, counts = np.unique(y_pred_classes, return_counts=True)
    for label, count in zip(classes, counts):
        print('class', label, ':', count)

    print('\nTest set metrics:')
    print(classification_report(y_test, y_pred_classes, digits=4))
    rep = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    report.to_csv(save_dir+'/'+direction+'_trfc_report_.csv')

    return

def gbrfc(direction: str, x_train: any, y_train: any, x_test: any, y_test: any):
    '''
    Training with sklearn gradient boosted random forest classifier. Supports
    grid search with GridSearchCV object.
    Unused in final thesis paper.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''
    from sklearn.model_selection import GridSearchCV

    if direction == 'outbound':
        params = {}
        print('\n========== Outbound Random Forest Classifier ==========')
        params = {'max_iter': 100,
                  'max_depth': 800,
                  'min_samples_leaf': 10,
                  'max_leaf_nodes':31,
                  'max_bins':255,
                  'sample_method':'smote',
                  'sample_ratio':1.0
                  }
        ''' 
        print('Grid Search')
        params = {'sample_method': 'smote',
                  'sample_ratio': 0.12
                  }
        grid = {'max_depth': [700, 1000],
                  'min_samples_split': [10, 100, 1000],
                  'min_samples_leaf': [10, 100, 1000]
                  }
        clf = GridSearchCV(RandomForestClassifier(warm_start=True), grid, verbose=3)
        '''
        
    elif direction == 'inbound':
        print('\n========== Inbound Random Forest Classifier =========')
        params = {'max_iter': 200,
                  'max_depth': 1000,
                  'min_samples_leaf': 100,
                  'max_leaf_nodes':100,
                  'max_bins':255,
                  'sample_method':'smote',
                  'sample_ratio':0.12
                  }
       
        ''' 
        print('Grid Search')
        params = {'sample_method': 'smote',
                  'sample_ratio': 1.0
                 }
        grid = {'max_depth': list(range(1000, 2000, 500)),
                  'n_estimators': [100],
                  'min_samples_split': [10, 50, 100, 1000],
                  'min_samples_leaf': [10, 50, 100, 1000]
                  }
        clf = GridSearchCV(RandomForestClassifier(warm_start=True), grid, verbose=3)
        '''
    else:
        raise Exception('direction not specified')

    x_train, y_train = sampling(x_train, y_train, params)
    
    clf = M.get_gbrfc_model(params)

    print('fitting ' + direction + ' classifier...')
    sys.stdout.flush()
    clf.fit(x_train, y_train)
    print('...done')
    #print(clf.best_params_)
    sys.stdout.flush()

    date = str(datetime.now())

    JOB_ID = sys.argv[3]
    save_dir = JOB_ID + '_' + direction + '_gbrfc_run' + date
    os.makedirs(save_dir, exist_ok=True)

    print('dumping model')
    dump(clf, save_dir +'/'+ direction +'_gbrfc_'+date+'.joblib')

    '''
    with open(save_dir+'/params.json', 'w') as file:
        file.write(json.dumps(params))
    '''

    # prelim inference and metrics
    print('\nPreliminary inference and metrics:\n')
    y_pred_test = clf.predict(x_test)

    print('\nTest set metrics:')
    rep = classification_report(y_test, y_pred_test, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_test, digits=4))
    report.to_csv(save_dir+'/'+direction+'_rfc_report.csv')

    return

def rfc(direction: str, x_train: any, y_train: any, x_test: any, y_test: any):
    '''
    Training with sklearn random forest classifier. Supports
    grid search with GridSearchCV object.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    JOB_ID = sys.argv[3]
    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_rfc_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)

    if direction == 'outbound':
        params = {}
        print('\n========== Outbound Random Forest Classifier ==========')
        params = {'n_estimators': 100,
                  'max_depth': 800,
                  'min_samples_split': 10, 
                  'min_samples_leaf': 10,
                  'max_leaf_nodes': None,
                  'sample_method':'smote',
                  'sample_ratio':0.6
                 }
        ''' 
        print('Grid Search')
        params = {'sample_method': 'smote',
                  'sample_ratio': 0.12
                  }
        grid = {'max_depth': [700, 1000],
                  'min_samples_split': [10, 100, 1000],
                  'min_samples_leaf': [10, 100, 1000]
                  }
        clf = GridSearchCV(RandomForestClassifier(warm_start=True), grid, verbose=3)
        '''
        
    elif direction == 'inbound':
        print('\n========== Inbound Random Forest Classifier =========')
        params = {'n_estimators': 100,
                  'max_depth': 800,
                  'min_samples_split': 20,
                  'min_samples_leaf': 20,
                  'max_leaf_nodes': None,
                  'sample_method':'smote',
                  'sample_ratio': 0.45 
                 }
        
        ''' 
        print('Grid Search')

        params = {'sample_method': 'smote',
                  'sample_ratio': 0.3 
                 }

        grid = {'max_depth': [300, 700, 800],
                  'n_estimators': [100],
                  'max_leaf_nodes': [100],
                  'min_samples_split': [10, 20, 50],
                  'min_samples_leaf': [10, 20, 30]
                }

        clf = GridSearchCV(RandomForestClassifier(warm_start=True), grid, verbose=3)
        '''

    else:
        raise Exception('direction not specified')

    input_size = x_train.shape[1]
    params['input_size'] = input_size

    x_train, y_train = sampling(x_train, y_train, params)

    clf = M.get_rfc_model(params)

    print('fitting ' + direction + ' classifier...')
    sys.stdout.flush()
    clf.fit(x_train, y_train)
    print('...done')
    #print('Grid Search Best params:', clf.best_params_)
    sys.stdout.flush()

    print('dumping model')
    dump(clf, save_dir +'/'+ direction +'_rfc_.joblib')

    with open(save_dir+'/params.yml', 'w') as file:
        yaml.dump(params, file)

    # prelim inference and metrics
    print('\nPreliminary inference and metrics:\n')
    y_pred_test = clf.predict(x_test)

    print('Parameters:', params)
    print('\nTest set metrics:')
    rep = classification_report(y_test, y_pred_test, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_test, digits=4))
    report.to_csv(save_dir+'/'+direction+'_rfc_report.csv')

    '''
    # extract feature names from dataframe for feature importance below
    if direction == 'inbound': 
        print('\n========== Inbound Random Forest Classifier Results ==========')   
        df = pd.read_parquet('../data/inbound_data.parq')
    elif direction == 'outbound':
        print('\n========== Outbound Random Forest Classifier Results ==========')
        df = pd.read_parquet('../data/outbound_data.parq')
    cols = df.columns.tolist()
    cols.remove('packet_label')
    sys.stdout.flush()

    # Feature Importance (MDI)
    feature_importance = pd.Series(clf.feature_importances_, 
                                   index=cols).sort_values(ascending=False)

    with pd.option_context(             # display all dataframe columns
    'display.max_rows', 10, 
    'display.max_columns', None,
    'display.precision', 4):
        print(feature_importance)
    '''
    return

def dnn(direction: str, x_train: any, y_train: any, x_val: any, y_val: any, x_test: any, y_test: any):
    '''
    Training with TensorFlow/Keras sequential neural network model. 

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_val (array-like): Validation data features.

        y_val (array-like): Validation data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''

    JOB_ID = sys.argv[3]
    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_dnn_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)
    print('Save Directory: ', save_dir)

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

    elif direction == 'inbound':
        reg_strength = 1e-6
        params = {'n_neurons':512,
                  'n_drop':0.2,
                  'activation':'swish',
                  'eta':1e-6,
                  'regularizer':keras.regularizers.L2(reg_strength),
                  'b_size':8192,
                  'n_epochs': 100,
                  'sample_ratio': 0.3,
                  'sample_method':'smote'
                  }
        params['reg_strength'] = reg_strength 


    '''init bias
    count = list(Counter(y_train).values())
    print(count)
    num_allowed = count[0]
    num_blocked = count[1]
    print('num allowed:', num_allowed)
    print('num blocked:', num_blocked)
    #resampled_steps_per_epoch = np.ceil(2.0*num_allowed/params['b_size'])
    #print('steps per epoch:', resampled_steps_per_epoch)


    # init output bias to factor for class imbalance
    # bias = log(pos/neg)
    #init_bias = np.log(num_blocked/num_allowed)
    #params['init_bias'] = init_bias

    #softmax bias https://github.com/alexmolas/alexmolas.github.io/blob/master/docs/optimal-biases/Optimal%20biases.ipynb
    total = num_allowed + num_blocked
    params['class_0_bias'] = np.log(num_allowed/total)
    params['class_1_bias'] = np.log(num_blocked/total)
    '''
    
    # resample data
    x_train, y_train = sampling(x_train, y_train, params)

    # cast x to tensor
    x_train = tf.cast(x_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)

    # one hot y for softmax
    y_train = tf.one_hot(y_train, 2)
    y_val = tf.one_hot(y_val, 2)

    # store input shape
    input_size = x_train.shape[1]
    params['input_size'] = input_size
    sys.stdout.flush()

    # build model
    model = M.get_dnn_model(params)
    print(model.summary())
    sys.stdout.flush()

    print('----- '+direction+' traffic deterministic neural network training -----')
    sys.stdout.flush()

    n_epochs = params['n_epochs']
    print('Total number of epochs:', n_epochs)
    sys.stdout.flush()

    history = model.fit(x_train, y_train,
                        epochs=n_epochs,
                        #steps_per_epoch=resampled_steps_per_epoch,
                        validation_data=(x_val, y_val),
                        batch_size=params['b_size'],
                        verbose=2,
                        callbacks=callbacks.make_callbacks(save_dir))

    # write params to params.txt
    print('Parameters:')
    params['regularizer'] = str(params['regularizer'])

    for p in params:
        print(p, ':', params[p])

    # save params dictionary to yaml file
    with open(save_dir + 'params.yml', 'w') as file:
        yaml.dump(params, file)

    # save metrics history
    print('Saving model history....')
    dump(history, save_dir + direction +'_dnn_history')

    # plot training curves
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title(direction + ' dnn loss')
    plt.ylabel('crossentropy loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.savefig(save_dir + direction+'_dnn_loss.png', bbox_inches='tight')

    '''
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.grid(True)
    plt.title(direction + ' dnn metrics')
    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall'], loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.savefig(save_dir + direction+'_dnn_metrics.png', bbox_inches='tight')
    '''

    print('================== ' + direction + ' model results ==================')
    y_pred = model.predict(x_test, verbose=2)

    # convert sigmoid probabilities into classes with thresh = 0.5
    #y_pred_proba = y_pred.flatten()
    #y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)
    # convert softmax probabilities
    print('y_pred')
    print(y_pred)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    print('Test Data Results:')
    print('y_test', y_test)
    print('true classes:')
    print(sorted(Counter(y_test).items()))

    y_pred_classes = y_pred_classes.numpy()
    print('y_pred', y_pred_classes)
    print('pred classes:')
    print(sorted(Counter(y_pred_classes).items()))

    rep = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_classes, digits=4))
    report.to_csv(save_dir + direction+'_dnn_report.csv')
    sys.stdout.flush()

    return
        
def tuner_dnn(direction: str, x_train: any, y_train: any, x_val: any, y_val: any, x_test: any, y_test: any):
    '''
    Training with TensorFlow/Keras sequential neural network model, performs grid search of
    hyperparameters using Keras tuner and hypermodel objects. 

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_val (array-like): Validation data features.

        y_val (array-like): Validation data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''
    date = str(datetime.now())
    sys.stdout.flush()
    save_dir = direction+'_tuner_dnn_run_'+date
    os.makedirs(save_dir, exist_ok=True) 
    
    params = {'sample_method': 'smote',
              'sample_ratio': 1.0
              }
    x_train, y_train = sampling(x_train, y_train, params)
    
    print('\ndnn input counts:')
    print(sorted(Counter(y_train).items()))

    input_size = x_train.shape[1]
    print('\nInput Size:', input_size)
    params['input_size'] = input_size
    sys.stdout.flush()

    # init keras tuner
    #hp = keras_tuner.HyperParameters()
    # FIXME tuner
    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)
    y_val = tf.cast(y_val, tf.float32)
    tuner = keras_tuner.GridSearch(
        hypermodel=M.dnn_hypermodel(params),
        objective=keras_tuner.Objective('val_loss', 'min'),
        directory='tuner',
        project_name=direction + '_dnn_' + str(datetime.now())
    )
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), verbose=2)
    models = tuner.get_best_models(num_models=2)
    model = models[0]
    model.summary()
    model.build()
    tuner.results_summary()

    print('Saving '+direction+' tuner dnn model...')


    model_path = save_dir + '/' + direction +'_dnn_model.keras'
    model.save(model_path)
    print('Save complete.')
    print('Model path:', model_path)
    sys.stdout.flush()    

    # save best params
    trials = tuner.oracle.get_best_trials(num_trials=2)
    HP_list = []
    for trial in trials:
        HP_list.append(trial.hyperparameters.get_config()["values"] | {"Score": trial.score})
    HP_df = pd.DataFrame(HP_list)
    HP_df.to_csv("name.csv", index=False, na_rep='NaN')


    print('================== ' + direction + ' model results ==================')
    y_pred = model.predict(x_test, verbose=2)

    y_pred_proba = y_pred.flatten()
    # convert sigmoid probabilities into classes with thresh = 0.5
    y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)

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
    report.to_csv(save_dir+'/'+direction+'_dnn_report.csv')
    sys.stdout.flush()

    return

def mcd_dnn(direction: str, x_test: any, y_test: any):
    '''
    Performs Monte Carlo dropout variational inference model testing by loading pre-trained DNN model and
    setting dropout training to True at test time.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None
    '''

    SAVE_DIR = '/smallwork/alexander.huang/mcdropout/'
    JOB_ID = sys.argv[3]
    RUN_DIR = SAVE_DIR + '/' + direction + '/' + JOB_ID + '/'
    os.makedirs(RUN_DIR, exist_ok=True)

    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_mcdnn_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)
    print('Save Directory: ', save_dir)

    if direction == 'outbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/checkpoint97-0.06.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/params.yml'
    elif direction == 'inbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/checkpoint82-0.14.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595421_inbound_dnn_run_2023-11-18 15:08:16.236068/params.yml'
    else:
        raise ValueError('data direction invalid, recieved:', direction)

    # load dnn model params
    with open(DNN_PARAMS_PATH) as file:
        params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # init dnn model with params
    model = M.get_dnn_model(params)

    # load dnn model weights
    model.load_weights(DNN_MODEL_PATH)

    n_models = 20 
    print('\nn samples:', n_models)
    sys.stdout.flush()

    if direction == 'inbound':
        print('================== inbound model results ==================')
    elif direction == 'outbound':
        print('================== outbound model results ==================')
    
    # inference with dropout at training and test time
    print('MCD inference')
    y_probas = np.stack([model(x_test, training=True) for _ in range(n_models)])

    print('MCD shape:', y_probas.shape)
    y_proba = y_probas.mean(axis=0)
    print('MCD mean shape:', y_proba.shape)
    y_pred_mcd = np.argmax(y_proba, axis=1)
    print('MCD predictions shape:', y_pred_mcd.shape)

    print('deterministic inference')
    y_pred_test = model.predict(x_test, verbose=2)
    y_pred_dnn = np.argmax(y_pred_test, axis=1)

    print('Dropout model test predictions:\n', y_pred_dnn)
    print('Monte Carlo Dropout test preductions:\n', y_pred_mcd)
    sys.stdout.flush

    print('Test Data Results for dropout model:')
    print(classification_report(y_test, y_pred_dnn, digits=4))

    print('\nTest Data Results for mc dropout model:')
    rep = classification_report(y_test, y_pred_mcd, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_mcd, digits=4))
    report.to_csv(save_dir + direction+'_dnn_report.csv')

    print('saving mcd results')
    np.save(RUN_DIR + 'mcd_predictions.npy', y_probas)
    np.save(RUN_DIR + 'dnn_predictions.npy', y_pred_test)
    print('done saving.')
    sys.stdout.flush()

    return

def bnn_vi(direction: str, x_train: any, y_train: any, x_val: any, y_val: any, x_test: any, y_test: any):
    '''
    Training with Gaussian prior variational inference model. Initializes posterior weights
    with weights from loaded DNN model.
    Unused in final thesis paper.

    Args:
    direction (str): "inbound" or "outbound", direction of network traffic dataset.

    x_train (array-like): Training data features.

    y_train (array-like): Training data labels.

    x_val (array-like): Validation data features.

    y_val (array-like): Validation data labels.

    x_test (array-like): Test data features.

    y_test (array-like): Test data labels.

    Returns:
        None

    Raises:
        ValueError: If direction is not "inbound" or "outbound."
    '''
    JOB_ID = sys.argv[3]
    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_vidnn_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)
    print('Save Directory: ', save_dir)

    params = {}
    if direction == 'outbound':
        params = {'n_neurons':512,
                  'eta':1e-5,
                  'prior_scale': 1.0,
                  'posterior_scale':0.1,
                  'b_size': 4096,
                  'n_epochs': 400, 
                  'sample_ratio': 1.0,
                  'sample_method':'smote'
                  }

    elif direction == 'inbound':
        params = {'n_neurons':512,
                  'eta':1e-5,
                  'b_size':8192,
                  'n_epochs': 800,
                  'sample_ratio':0.1,
                  'sample_method':'smote'
                  }
        
    input_size = x_train.shape[1]
    print('input size:', input_size) 
    params['input_size'] = input_size
    sys.stdout.flush()

    n_examples = y_train.shape[0]
    print('number of examples:', n_examples)
    params['n_examples'] = n_examples
    sys.stdout.flush()

    ''' output bias
    count = list(Counter(y_train).values())
    print(count)
    num_allowed = count[0]
    num_blocked = count[1]

    # init output bias to factor for class imbalance
    output_bias = np.log([num_blocked/num_allowed])
    params['output_bias'] = output_bias 
    '''

    # i hate tensorflow
    x_train = tf.cast(x_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)
    y_train = tf.one_hot(y_train, 2)
    y_val = tf.one_hot(y_val, 2)

    print('loading dnn model weights...')
    sys.stdout.flush()

    # tf 2.13 compat dnn model, weights only
    if direction == 'outbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-13/49111756_outbound_dnn_run_2023-10-13 13:27:28.481509/checkpoint100-0.06.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-13/49111756_outbound_dnn_run_2023-10-13 13:27:28.481509/params.yml'
    elif direction == 'inbound':
        DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-10-16/49112464_inbound_dnn_run_2023-10-16 12:58:36.901380/checkpoint100-0.14.h5'
        DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-10-16/49112464_inbound_dnn_run_2023-10-16 12:58:36.901380/params.yml'
    else:
        raise ValueError('Invalid dataset, recieved:', direction)

    params['dnn_model_path'] = DNN_MODEL_PATH
    params['dnn_params_path'] = DNN_PARAMS_PATH

    sys.stdout.flush()

    model = M.get_vi_model(params)

    print(model.summary())
    sys.stdout.flush()

    print('----- '+direction+' traffic deterministic neural network training -----')
    sys.stdout.flush()

    n_epochs = params['n_epochs']
    print('Total number of epochs:', n_epochs)
    sys.stdout.flush()
    history = model.fit(x_train, y_train,
                        epochs=n_epochs,
                        validation_data=(x_val, y_val),
                        batch_size=params['b_size'],
                        verbose=2,
                        callbacks=callbacks.make_callbacks(save_dir))

    # write params to params.txt
    print('Parameters:')
    for p in params:
        print(p, ':', params[p])

    with open(save_dir+'/params.yml', 'w') as file:
        yaml.dump(params, file)

    # save model
    # save metrics history
    print('Saving model history....')
    dump(history, save_dir+'/'+direction +'_vinn_history.history')

    # plot training curves
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title(direction + ' dnn loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_dir+'/'+direction+'_vinn_loss.png', bbox_inches='tight')

    '''
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.grid(True)
    plt.title(direction + ' dnn metrics')
    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall'], loc='upper left')
    plt.savefig(save_dir+'/'+direction+'_dnn_metrics.png', bbox_inches='tight')
    '''

    print('================== ' + direction + ' model results ==================')
    # flatten output
    #y_pred_proba = y_pred.flatten()
    # convert sigmoid probabilities into classes with thresh = 0.5
    #y_pred_classes = np.where(y_pred_proba > 0.5, 1 , 0)
    del x_train
    del y_train 
    collect()

    ''' validate output distribution
    yhat = model(x_test)
    assert(isinstance(yhat, tfd.Distribution))
    del yhat
    collect()

    n_models = 20 
    y_preds = []
    for i in range(n_models):
        y_pred = model.predict(x_test, verbose=2)
        y_preds.append[y_pred]
    '''
    
    y_pred = model.predict(x_test, verbose=2)

    #y_proba = y_probas.mean(axis=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print('Test Data Results:')
    print('y_test')
    print(y_test)
    print('true classes:')
    classes, counts = np.unique(y_test, return_counts=True)
    for label, count in zip(classes, counts):
        print('class', label, ':', count)
    print('y_pred')
    print(y_pred_classes)
    print('pred classes:')
    classes, counts = np.unique(y_pred_classes, return_counts=True)
    for label, count in zip(classes, counts):
        print('class', label, ':', count)
    print(classification_report(y_test, y_pred_classes, digits=4))
    sys.stdout.flush()

    return

def concrete_dropout_nn(direction: str, x_train: any, y_train: any, x_val: any, y_val: any, x_test: any, y_test: any):
    '''
    Training using concrete dropout variational inference by training
    a Keras sequential model with concrete dropout layers in lieu of standard
    dropout layers.

    Args:
        direction (str): "inbound" or "outbound", direction of network traffic dataset.

        x_train (array-like): Training data features.

        y_train (array-like): Training data labels.

        x_val (array-like): Validation data features.

        y_val (array-like): Validation data labels.

        x_test (array-like): Test data features.

        y_test (array-like): Test data labels.

    Returns:
        None

    '''

    SAVE_DIR = '/smallwork/alexander.huang/concrete_dropout/'
    DIRECTION = sys.argv[1]
    JOB_ID = sys.argv[3]
    RUN_DIR = SAVE_DIR + '/' + DIRECTION + '/' + JOB_ID + '/'
    os.makedirs(RUN_DIR, exist_ok=True)

    time = str(datetime.now())
    save_dir = '/smallwork/alexander.huang/models/' + str(date.today()) + '/' + JOB_ID + '_' + direction+'_cdnn_run_'+ time + '/'
    os.makedirs(save_dir, exist_ok=True)
    print('Save Directory: ', save_dir)

    params = {}
    if direction == 'outbound':
        # parameters
        params = {'n_neurons':512,
                  'weight_reg': 1e-4,
                  'activation':'swish',
                  'eta':1e-6,
                  'b_size': 4096,
                  'n_epochs': 200, 
                  'sample_ratio': 0.1,
                  'sample_method':'smote',
                  }

    elif direction == 'inbound':
        params = {'n_neurons':512,
                  'weight_reg': 1e-6,
                  'activation':'swish',
                  'eta':1e-6,
                  'b_size':8192,
                  'n_epochs': 300,
                  'sample_ratio': 0.3,
                  'sample_method':'smote',
                  }

    # resample data
    x_train, y_train = sampling(x_train, y_train, params)

    x_train = tf.cast(x_train, tf.float32)
    x_val = tf.cast(x_val, tf.float32)

    # one hot for softmax
    y_train = tf.one_hot(y_train, 2)
    y_val = tf.one_hot(y_val, 2)

    input_size = x_train.shape[1]
    params['input_size'] = input_size
    params['n_examples'] = x_train.shape[0]
    sys.stdout.flush()

    model = M.get_cdnn_model(params)

    print(model.summary())
    sys.stdout.flush()

    print('----- '+direction+' traffic concrete dropout neural network training -----')
    sys.stdout.flush()

    n_epochs = params['n_epochs']
    print('Total number of epochs:', n_epochs)
    sys.stdout.flush()

    history = model.fit(x_train, y_train,
                        epochs=n_epochs,
                        validation_data=(x_val, y_val),
                        batch_size=params['b_size'],
                        verbose=2,
                        callbacks=callbacks.make_callbacks(save_dir))

    # write params to params.txt
    print('Parameters:')
    for p in params:
        print(p, ':', params[p])

    # save params dictionary to yaml file
    with open(save_dir + 'params.yml', 'w') as file:
        yaml.dump(params, file)

    # save metrics history
    print('Saving model history....')
    dump(history, save_dir + direction +'_cdnn_history')

    # plot training curves
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title(direction + ' dnn loss')
    plt.ylabel('crossentropy loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.savefig(save_dir + direction+'_cdnn_loss.png', bbox_inches='tight')

    print('================== ' + direction + ' model results ==================')
    # preserve memory 
    del x_train
    del y_train
    del x_val
    del y_val
    collect()

    n_models = 1
    print('single model inference w/ Concrete Dropout')
    y_pred = model.predict(x_test, verbose=2)

    y_pred_classes = np.argmax(y_pred, axis=1)
    print('class predictions shape:', y_pred_classes.shape)

    print('test predictions:\n', y_pred_classes)
    sys.stdout.flush

    ps = np.array([keras.backend.eval(layer.p_logit) for layer in model.layers if hasattr(layer, 'p_logit')])
    dropout_val = tf.nn.sigmoid(ps).numpy()
    print('Learned Dropout Values:', dropout_val)

    print('\nTest Data Results for concrete dropout model:')
    rep = classification_report(y_test, y_pred_classes, digits=4, output_dict=True)
    report = pd.DataFrame(rep).transpose()
    print(classification_report(y_test, y_pred_classes, digits=4))
    report.to_csv(save_dir + direction+'_dnn_report.csv')
    sys.stdout.flush()

    print('done saving.')

    return

if __name__ == '__main__':

    direction = sys.argv[1]
    model = sys.argv[2] 
    print('data:', DATA_DIR)

    if len(sys.argv) != 4:
        raise Exception('invalid input arguments, expected 3 arguments, recieved:'+str(sys.argv))
    else:
        print('*** run parameters: ***')
        print('direction:', direction)
        print('model:', model)
        print('data:', DATA_DIR)

        # load data from file
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(direction)

        ''' unused logistic regression model code, kept for reference.
        # logistic regression model
        if model == 'log':
            pass
            #log_reg_model(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        elif model == 'log_vi':
            pass
            #log_vi_model(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        ''' 
        if model == 'rfc':
            rfc(direction, x_train, y_train, x_test, y_test)
        elif model == 'gbrfc':
            gbrfc(direction, x_train, y_train, x_test, y_test)
        elif model =='trfc':
            tf_rfc(direction, x_train, y_train, x_test, y_test)
        elif model == 'tuner_dnn':
            tuner_dnn(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        elif model == 'dnn':
            dnn(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        elif model == 'mcd':
            mcd_dnn(direction, x_test, y_test)
        elif model == 'nn_vi':
            bnn_vi(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        elif model == 'cdnn':
            concrete_dropout_nn(direction, x_train, y_train, x_val, y_val, x_test, y_test)
        else:
            raise Exception('Error: invalid model option.')

    end_time = datetime.now()
    total_time = datetime.now() - start_time

    # print runtime information
    print('Start time:', start_time)
    print('End time:', end_time)
    print('Elapsed:', total_time)

    sys.exit(0)
