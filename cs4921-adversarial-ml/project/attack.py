import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from tensorflow import keras

sys.path.insert(0, '/home/alexander.huang/repositories/huang-thesis/models')
from train import load_data
import models as M
import yaml

from sklearn.metrics import classification_report


def get_bounds(target_feature, feature_option, df):
    '''
    Get array boundaries of target feature.

    Params: 
        target_feature: String for target feature name or list of strings for geolocation features

    Returns:
        target_bounds: Tuple of indices for target feature
    '''
    first = False
    a = 0
    b = 0

    if type(target_feature) is list:
        for i in range(len(df.columns)):
            if df.columns[i].startswith(target_feature[0]) or df.columns[i].startswith(target_feature[1]):
                if first == False:
                    a = i
                    first = True
                b = i+1
                print(f'{i}: {df.columns[i]}')

    elif type(target_feature) is str:
        if feature_option == 'cc':
            for i in range(len(df.columns)):
                if len(df.columns[i]) == 16:
                    if first == False:
                        a = i
                        first = True
                    b = i+1
                    print(f'{i}: {df.columns[i]}')
        elif feature_option == 'port':
            for i in range(len(df.columns)):
                if df.columns[i].startswith(target_feature):
                    if first == False:
                        a = i
                        first = True
                    b = i+1
                    print(f'{i}: {df.columns[i]}')
    else:
        raise ValueError(f'target feature must be a string or list of strings, received: {target_feature}')

    target_bounds = (a, b)
    print('target bounds:', target_bounds)
    return target_bounds


def gen_adv(model: tf.keras.Model, x_set, y_set, target_class: int, target_bounds: tuple, lr: float, steps: int, temp: float) -> int:
    '''
    Generates adversarial one-hot encoded array for feature within target bounds.

    Returns:
        adv_feature_index: Feature index to set for adversarial attack.
         
    '''
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    a, b = target_bounds        

    # target blocked -> allowed
    if target_class == 0:
        #blocked_index = np.where(y_pred_classes == 1)
        blocked_index = np.where(y_set == 1)
        x_block = np.take(x_set, blocked_index, axis=0)
        y_block = np.take(y_set, blocked_index, axis=0)
        x = np.squeeze(x_block)
        y = np.squeeze(y_block)

    # target allowed -> blocked 
    elif target_class == 1:
        allow_index = np.where(y_set == 0)
        x_allow = np.take(x_set, allow_index, axis=0)
        y_allow = np.take(y_set, allow_index, axis=0)
        x = np.squeeze(x_allow)
        y = np.squeeze(y_allow)

    elif target_class == None:
        x = x_set
        y = y_set

    print(x.shape)
    print(y.shape)
    # separate other features from target features
    x_left= x[:,:a]
    x_right= x[:,b:]
    # target features are z, separated for grad descent below
    z = tf.zeros_like(x[0,a:b])
    z = tf.reshape(z, [1,-1])
    # setup z and y for grad descent
    z = tf.Variable(z, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, 2, dtype=tf.float32)
    
    for _ in tqdm(range(steps)):
        # gradients of z wrt loss
        with tf.GradientTape() as t1:
            t1.watch(z)
            # apply softmax with low temperature
            z1 = tf.nn.softmax(z/temp, axis=1)

            # ensure z is compatible shape with x
            z_add = tf.repeat(z1, x_left.shape[0], axis=0)
            # combine x and z 
            x = tf.concat([x_left, z_add], axis=1)
            x = tf.concat([x, x_right], axis=1)

            y_prob = model(x, training=True)
            loss_value = loss_fn(y, y_prob)

        # get gradients, reverse the grad for maximizing loss for opposing class 
        grad_z = -t1.gradient(loss_value, z)

        opt.apply_gradients(zip([grad_z], [z]))

    #z_softmax = tf.nn.softmax(grad_z/temp, axis=1)
    adv_feature_index = np.argmax(z.numpy().squeeze())

    # return index of adversarial feature within z array
    # set this index to 1 in corresponding feature array in input x for evasion attack
    return adv_feature_index


def gen_adv_geo(model: tf.keras.Model, x_set, y_set, target_class: int, target_bounds: tuple, lr: float, steps: int) -> tuple:
    '''
    Generates adversarial feature values for lat/long feature within target bounds.

    Returns:
        adv_feature_index: Tuple of feature values to set for adversarial attack.
         
    '''
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # should just be lat and long, array of size 2
    a, b = target_bounds        

    # untargeted, take all samples
    x = x_set
    y = y_set
        
    print(x.shape)
    print(y.shape)
    # separate other features from target features
    x_left= x[:,:a]
    x_right = x[:,b:]

    # target features are z, separated for grad descent below
    z = x[0,a:b]
    z = tf.reshape(z, [1,-1])

    # setup z and y for grad descent
    z = tf.Variable(z, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, 2, dtype=tf.float32)
    
    for _ in tqdm(range(steps)):
        # gradients of z wrt loss
        with tf.GradientTape() as t1:
            # ensure z is compatible shape with x
            z_add = tf.repeat(z, x_left.shape[0], axis=0)
            # combine x_left + z + x_right 
            x = tf.concat([x_left, z_add], axis=1)
            x = tf.concat([x, x_right], axis=1)

            y_prob = model(x, training=True)
            loss_value = loss_fn(y, y_prob)

        # get gradients, reverse the grad for maximizing loss for opposing class 
        if target_class == None or target_class == 1:
            grad_z = -t1.gradient(loss_value, z)
        elif target_class == 0:
            grad_z = t1.gradient(loss_value, z)

        opt.apply_gradients(zip([grad_z], [z]))

    grad_z = tf.squeeze(grad_z)

    # clip values to lat/long
    # lat first, then long
    latitude = tf.clip_by_value(grad_z[0], -90, 90).numpy().squeeze()
    longitude = tf.clip_by_value(grad_z[1], -180, 180).numpy().squeeze()

    # return lat/long coordinates to bypass defense
    print(f'Found adv location coordinates: \nLatitude:{latitude}\nLongitude:{longitude}')
    return latitude, longitude 


def test_port(model, index_list, x_test, target_bounds, index):
    '''
    Test script for success rate.
    '''
    count = 0
    total = len(index_list)
    a, b = target_bounds
    for i in tqdm(index_list):
        test_sample = np.copy(x_test[i])
        # zero out z in test sample
        test_sample[a:b] = 0
        # make adversarial z vector
        noise = np.zeros(shape=(b-a))
        noise[index] = 1
        # replace z with adversarial z
        test_sample[a:b] += noise
        # test original x and adversarial x+z
        adv_pred = np.argmax(model.predict(np.reshape(test_sample, [1,-1]), verbose=None), axis=1).squeeze()
        orig_pred = np.argmax(model.predict(np.reshape(x_test[i], [1,-1]), verbose=None), axis=1).squeeze()
        #print(f'Example {i} - ')
        #print(f'Original prediction: {orig_pred}')
        #print(f'Adv prediction: {adv_pred}')
        if adv_pred != orig_pred:
            count += 1
    print(f'Success rate: {round(count/total*100, 2)}%')


def test_cc(model, index_list, x_test, target_bounds, index):
    '''
    Test script for success rate.
    '''
    count = 0
    total = len(index_list)
    a, b = target_bounds
    for i in tqdm(index_list):
        test_sample = np.copy(x_test[i])
        # zero out z in test sample
        test_sample[a:b] = 0
        # make adversarial z vector
        noise = np.zeros(shape=(b-a))
        noise[index] = 1
        # replace z with adversarial z
        test_sample[a:b] += noise
        # test original x and adversarial x+z
        adv_pred = np.argmax(model.predict(np.reshape(test_sample, [1,-1]), verbose=None), axis=1).squeeze()
        orig_pred = np.argmax(model.predict(np.reshape(x_test[i], [1,-1]), verbose=None), axis=1).squeeze()
        #print(f'Example {i} - ')
        #print(f'Original prediction: {orig_pred}')
        #print(f'Adv prediction: {adv_pred}')
        if adv_pred != orig_pred:
            count += 1
    print(f'Success rate: {round(count/total*100, 2)}%')


def test_geo(model, index_list, x_test, target_bounds, loc):
    '''
    Test script for success rate.
    '''
    print('test loc:', loc)
    count = 0
    total = len(index_list)
    a, b = target_bounds
    for i in tqdm(index_list):
        test_sample = np.copy(x_test[i])
        # zero out z in test sample
        test_sample[a:b] = 0
        # make adversarial z vector
        noise = np.zeros(shape=(b-a))
        noise = loc
        # replace z with adversarial z
        test_sample[a:b] += noise
        # test original x and adversarial x+z
        adv_pred = np.argmax(model.predict(np.reshape(test_sample, [1,-1]), verbose=None), axis=1).squeeze()
        orig_pred = np.argmax(model.predict(np.reshape(x_test[i], [1,-1]), verbose=None), axis=1).squeeze()
        #print(f'Example {i} - ')
        #print(f'Original prediction: {orig_pred}')
        #print(f'Adv prediction: {adv_pred}')
        if adv_pred != orig_pred:
            count += 1
    print(f'Success rate: {round(count/total*100, 2)}%')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ValueError(f'Incorrect arguments, received: {sys.argv}.' \
                         f'\nUsage: {sys.argv[0]} <attack feature>' \
                         f'Allowed attack feature options: \n"port" for port feature\n"geo" for lat/long features')

    # feature to attack
    # 'port' for port feature
    # 'geo' for lat/long features
    FEATURE = sys.argv[1]

    # read datasets
    x_train, y_train, x_val, y_val, x_test, y_test = load_data('outbound')

    # load model
    DNN_MODEL_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/checkpoint97-0.06.h5'
    DNN_PARAMS_PATH = '/smallwork/alexander.huang/models/2023-11-18/49595396_outbound_dnn_run_2023-11-18 14:21:45.406099/params.yml'

    with open(DNN_PARAMS_PATH) as file:
        params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # init dnn model with params
    model = M.get_dnn_model(params)

    # load dnn model weights
    model.load_weights(DNN_MODEL_PATH)

    # check model
    model.summary()
    # get original test predictions
    y_pred = model.predict(x_test, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)
    #print(classification_report(y_test, y_pred_classes, digits=4))

    # read dataframe to obtain feature names
    DATA_DIR = '/data/alexander.huang/data/0903_data/test_drop/outbound_x_test.parq'
    df = pd.read_parquet(DATA_DIR)


    print('\n----------------------------- Attack Setup ----------------------------')
    target_dict = {}

    if FEATURE == 'port':
        # get bounds of targeted feature vector
        target_feature = 'inside_port_dst'
        print('\nTargeted Feature:', target_feature)
        target_bounds = get_bounds(target_feature, FEATURE, df)
        target_dict['port'] = target_bounds

        # get target feature names
        features = []
        for i in range(len(df.columns)):
            if df.columns[i].startswith(target_feature):
                features.append(df.columns[i])

    elif FEATURE == 'cc':
        # get bounds of targeted feature vector
        target_feature = 'inside_ip_dst'
        print('\nTargeted Feature:', target_feature)
        target_bounds = get_bounds(target_feature, FEATURE, df)
        target_dict['cc'] = target_bounds

        # get target feature names
        features = []
        for i in range(len(df.columns)):
            if df.columns[i].startswith(target_feature):
                features.append(df.columns[i])

        print('features:', features)

    elif FEATURE == 'geo':
        target_features = ['inside_ip_dst_LAT', 'inside_ip_dst_LONG']
        print('\nTargeted Feature:', target_features)
        target_bounds = get_bounds(target_features, FEATURE, df)
        target_dict['geo'] = target_bounds

    elif FEATURE == 'both':
        # get bounds of targeted feature vector
        target_feature = 'inside_port_dst'
        print('\nTargeted Feature:', target_feature)
        target_bounds = get_bounds(target_feature, df)
        target_dict['port'] = target_bounds

        # get target feature names
        features = []
        for i in range(len(df.columns)):
            if df.columns[i].startswith(target_feature):
                features.append(df.columns[i])

        target_features = ['inside_ip_dst_LAT', 'inside_ip_dst_LONG']
        print('\nTargeted Feature:', target_feature)
        target_bounds = get_bounds(target_feature, df)
        target_dict['geo'] = target_bounds

    for ele in target_dict:
        print(f'{ele}: {target_dict[ele]}')

    # setup adv attack function
    # learning rate for grad descent evasion attack
    LR = 1e-5 
    # number of grad descent steps
    STEPS = 1000
    # softmax temperature, use a very small value
    TEMP = 1e-5
    # None for untargeted attack
    TARGET = 1 

    if TARGET == None:
        print('\n---------------------- Untargeted attack ----------------------------')
        sys.stdout.flush()
    else:
        print('\n---------------------- Targeted:', TARGET, '-------------------------')
        sys.stdout.flush()

    # number of validation set samples to use in attack
    SAMPLES = 1000
    TEST_SAMPLES = 10000
    if FEATURE == 'port':
        adv_index = gen_adv(model=model,
                            x_set=x_train[:SAMPLES],
                            y_set=y_train[:SAMPLES],
                            target_class=TARGET,
                            target_bounds=target_dict['port'], 
                            lr=LR, 
                            steps=STEPS, 
                            temp=TEMP
                            )

        print('\nAdversarial feature index:', adv_index)
        sys.stdout.flush()
        print('\nAdversarial feature name:', features[adv_index])
        sys.stdout.flush()

        # FIXME: test on full test set
        block_index = np.where(y_pred_classes == 1)
        allow_index = np.where(y_pred_classes == 0)

        print('=============================== Testing ==============================')
        print('\nBlocked class:')
        # only test samples where original prediction was blocked
        test_port(model, block_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_index)
        sys.stdout.flush()

        print('\nAllowed class:')
        # only test samples where original prediction was allowed
        test_port(model, allow_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_index)
        sys.stdout.flush()

    elif FEATURE == 'cc':
        adv_index = gen_adv(model=model,
                             x_set=x_val[:SAMPLES],
                             y_set=y_val[:SAMPLES],
                             target_class=TARGET,
                             target_bounds=target_dict['cc'], 
                             lr=LR, 
                             steps=STEPS, 
                             temp=TEMP
                             )

        print('\nAdversarial feature index:', adv_index)
        sys.stdout.flush()
        print('\nAdversarial feature name:', features[adv_index])
        sys.stdout.flush()

        # FIXME: test on full test set
        block_index = np.where(y_pred_classes == 1)
        allow_index = np.where(y_pred_classes == 0)

        print('=============================== Testing ==============================')
        print('\nBlocked class:')
        # only test samples where original prediction was blocked
        test_cc(model, block_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_index)
        sys.stdout.flush()

        print('\nAllowed class:')
        # only test samples where original prediction was allowed
        test_cc(model, allow_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_index)
        sys.stdout.flush()
    
    elif FEATURE == 'geo':
        adv_loc = gen_adv_geo(model=model,
                              x_set=x_val[:SAMPLES],
                              y_set=y_val[:SAMPLES],
                              target_class=TARGET,
                              target_bounds=target_dict['geo'], 
                              lr=LR, 
                              steps=STEPS, 
                              )

        print('\nAdversarial coordinates:', adv_loc)
        sys.stdout.flush()

        # FIXME: test on full test set
        block_index = np.where(y_pred_classes == 1)
        allow_index = np.where(y_pred_classes == 0)

        print('=============================== Testing ==============================')
        print('\nBlocked class:')
        # only test samples where original prediction was blocked
        test_geo(model, block_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_loc)
        sys.stdout.flush()

        print('\nAllowed class:')
        # only test samples where original prediction was allowed
        test_geo(model, allow_index[0][:TEST_SAMPLES], x_test, target_bounds, adv_loc)
        sys.stdout.flush()