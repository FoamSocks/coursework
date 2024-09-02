from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_tuner
import yaml
import tensorflow_probability as tfp

from concrete_dropout import ConcreteDenseDropout
        
class dnn_hypermodel(keras_tuner.HyperModel):
    def __init__(self, params):
        self.params = params

    def dnn_model(self, units, reg, dropout, lr, metric):
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.params['input_size'])),
            keras.layers.Dense(units=units, 
                    activation='swish', 
                    kernel_regularizer=keras.regularizers.L1(reg), 
                    bias_regularizer=keras.regularizers.L1(reg)),
            keras.layers.Dropout(rate=dropout), #test
            keras.layers.Dense(units=units, 
                    activation='swish', 
                    kernel_regularizer=keras.regularizers.L1(reg), 
                    bias_regularizer=keras.regularizers.L1(reg)),
            keras.layers.Dropout(rate=dropout),
            keras.layers.Dense(2, activation='softmax', name='output')])

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(learning_rate=lr),
                      metrics=[metric])
        return model

    def build(self, hp):
        units = hp.Int('n_neurons', min_value=256, max_value=512, step=128)
        reg = hp.Float('reg_strength', min_value=0.0000001, max_value=0.01, step=10, sampling='log')
        drop = hp.Float('drop', min_value=0.2, max_value=0.5, step=0.1)
        lr = hp.Float('lr', min_value=1e-7, max_value=1e-5, step=10, sampling='log')
        metric = [keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall()]
        model = self.dnn_model(units, reg, drop, lr, metric) 
        return model
   
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
                         batch_size=hp.Choice('batch_size', [1024, 2048]),
                         epochs=200,
                         **kwargs)

def get_dnn_model(params):
    input = keras.layers.Input(shape=(params['input_size'],))
    h1 = keras.layers.Dense(params['n_neurons'], 
                    activation=params['activation'], 
                    kernel_regularizer=params['regularizer'], 
                    bias_regularizer=params['regularizer'], name='hidden_1')(input)
    d1 = keras.layers.Dropout(params['n_drop'])(h1)
    h2 = keras.layers.Dense(params['n_neurons'], 
                    activation=params['activation'], 
                    kernel_regularizer=params['regularizer'], 
                    bias_regularizer=params['regularizer'], name='hidden_2')(d1)
    d2 = keras.layers.Dropout(params['n_drop'])(h2)
    output = keras.layers.Dense(2, activation='softmax', name='output')(d2)
    
    model = keras.Model(inputs=input, outputs=output, name='dnn')

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(learning_rate=params['eta']),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model


def get_cdnn_model(params):
    # concrete dropout model
    n_examples = params['n_examples']

    #weight_reg = l**2/n_examples
    weight_reg = params['weight_reg'] 
    drop_reg = 1./n_examples

    input = keras.layers.Input(shape=(params['input_size'],))
    h1 = keras.layers.Dense(params['n_neurons'], 
                            activation=params['activation'], 
                            name='hidden_1')(input)
    h2 = keras.layers.Dense(params['n_neurons'], 
                            activation=params['activation'], 
                            name='hidden_2')
    d1 = ConcreteDenseDropout(h2, 
                              weight_regularizer=weight_reg, 
                              dropout_regularizer=drop_reg,
                              name='hidden_1_drop')(h1)
    output = keras.layers.Dense(2, activation='softmax', name='output')
    d2 = ConcreteDenseDropout(output, 
                              weight_regularizer=weight_reg, 
                              dropout_regularizer=drop_reg,
                              name='hidden_2_drop')(d1)
 
    model = keras.Model(inputs=input, outputs=d2, name='concrete_dropout_model')

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(learning_rate=params['eta']),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model


def get_stdnn_model(params):
    input = keras.layers.Input(params['input_size'])
    h1 = keras.layers.Dense(params['n_neurons'], 
                    activation=params['activation'], 
                    kernel_regularizer=params['regularizer'], 
                    bias_regularizer=params['regularizer'], name='hidden_1')(input)
    d1 = keras.layers.Dropout(params['n_drop'])(h1)
    h2 = keras.layers.Dense(params['n_neurons'], 
                    activation=params['activation'], 
                    kernel_regularizer=params['regularizer'], 
                    bias_regularizer=params['regularizer'], name='hidden_2')(d1)
    d2 = keras.layers.Dropout(params['n_drop'])(h2)

    temperature = 8 
    print('temperature:', temperature, flush=True)
    temp = keras.layers.Lambda(lambda x: x/temperature, name='temperature')(d2)

    output = keras.layers.Dense(2, activation='softmax', name='output')(temp)
    model = keras.Model(inputs=input, outputs=output, name='stdnn')

    @keras.saving.register_keras_serializable(package='loss_fn_pkg', name='weighted_ce_loss')
    def weighted_ce_loss(y_true, y_pred):
        cce = tf.losses.CategoricalCrossentropy()
        soft_weight = 1
        # use lower weight -> best results per Hinton paper
        hard_weight = .7 
        print('soft weight:', soft_weight, '- hard weight', hard_weight)
        # format of true labels: 0,1 is soft label, 2,3 is hard label
        # e.g. [[.32, .68, 0, 1]]
        y_true_soft = y_true[:,0:2] 
        y_true_hard = y_true[:,2:4] 

        # categorical cross-entropy between predictions and RFC soft labels
        soft_loss = cce(y_true_soft, y_pred)
        # categorical cross-entropy between predictions and ground truth hard labels
        hard_loss = cce(y_true_hard, y_pred)

        # weighted sum of cross-entropy loss of the soft labels and the hard labels
        total_loss = soft_weight * soft_loss + hard_weight * hard_loss

        return total_loss


    model.compile(loss=weighted_ce_loss,
                  optimizer=tf.optimizers.Adam(learning_rate=params['eta']),
                  #metrics=[keras.metrics.CategoricalAccuracy()]
                 )

    return model
    
def get_rfc_model(params):
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_leaf_nodes=params['max_leaf_nodes'],
        n_jobs=-1,
        warm_start=True,
        verbose=3,
        # for inbound class weights
        ##class_weight='balanced'
        )
    return model

def get_gbrfc_model(params):
    model = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        max_depth=params['max_depth'],
        max_bins=params['max_bins'],
        max_iter=params['max_iter'],
        min_samples_leaf=params['min_samples_leaf'],
        max_leaf_nodes=params['max_leaf_nodes'],
        l2_regularization=0,
        # categorical features are already OHE
        categorical_features=None,
        warm_start=True,
        early_stopping=True,
        verbose=3
    )
    return model

def get_vi_model(params):

    # load dnn params from yml
    print('loading dnn model from path:', params['dnn_model_path'])
    with open(params['dnn_params_path']) as file:
        dnn_params = yaml.load(file, Loader=yaml.loader.FullLoader)

    # create dnn model instance to init vi model
    dnn_model = get_dnn_model(dnn_params)
    dnn_model.load_weights(params['dnn_model_path'])

    #model.layer[index].get_weights()[0]
    theta = dnn_model.get_weights()
    print('theta:', len(theta))
    
    # get dnn hidden 1 weights
    l1_weights=theta[0]
    # get dnn hidden 2 weights
    l2_weights=theta[2]
    # get dnn hidden 1 biases
    l1_bias=theta[1]
    # get dnn hidden 2 biases
    l2_bias=theta[3]

    print('layer 1 weight shape:', l1_weights.shape)
    print('layer 1 bias shape:', l1_bias.shape)
    print('layer 2 weight shape:', l2_weights.shape)
    print('layer 2 bias shape:', l2_bias.shape)

    # bayesian model with NLL
    batch_size = params['b_size']
    n_examples = params['n_examples']

    def get_kernel_divergence_fn(train_size, kl_weight=1.0):
        def kernel_divergence_fn(q, p, _):
            kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
            return kl_weight * kernel_divergence
        return kernel_divergence_fn

    kl_div_fn = get_kernel_divergence_fn(n_examples)

    prior_scale = params['prior_scale'] 
    prior = tfp.layers.default_mean_field_normal_fn(
        # loc is mean, scale is std
        # init prior to a 0,1 mean field gaussian distribution
        is_singular=False,
        loc_initializer=tf.zeros_initializer(),
        untransformed_scale_initializer=tf.constant_initializer(prior_scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )

    scale = params['posterior_scale'] 
    print('prior scale', prior_scale)
    print('posterior scale:', scale)

    # 1. Iterate through DNN and BNN
    # 2. For each DNN layer, get_weights -> extract weight and bias mean from array
    # 3. For each BNN layer, init mean field gaussian with mean and variance from array
    ''' testing with default posteriors
    kernel_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=False,
        loc_initializer=tf.constant_initializer(0.0),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )

    bias_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=True,
        loc_initializer=tf.constant_initializer(0.0),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )
    '''

    l1_kernel_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=False,
        loc_initializer=tf.constant_initializer(l1_weights),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )

    l1_bias_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=True,
        loc_initializer=tf.constant_initializer(l1_bias),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )

    l2_kernel_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=False,
        loc_initializer=tf.constant_initializer(l2_weights),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )

    l2_bias_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=True,
        loc_initializer=tf.constant_initializer(l2_bias),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )


    '''
    if 'output_bias' not in params.keys():
        params['output_bias'] = 0

    output_bias_posterior =  tfp.layers.default_mean_field_normal_fn(
        is_singular=True,
        loc_initializer=tf.constant_initializer(params['output_bias']),
        untransformed_scale_initializer=tf.constant_initializer(scale),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None
    )
    '''

    input = keras.layers.Input(params['input_size'])
    h1 = tfp.layers.DenseFlipout(params['n_neurons'],
                                                 kernel_posterior_fn=l1_kernel_posterior,
                                                 bias_posterior_fn=l1_bias_posterior,
                                                 kernel_prior_fn=prior,
                                                 bias_prior_fn=prior,
                                                 kernel_divergence_fn=kl_div_fn,
                                                 bias_divergence_fn=kl_div_fn,
                                                 name='hidden_1')(input)
    h2 = tfp.layers.DenseFlipout(params['n_neurons'],
                                                 kernel_posterior_fn=l2_kernel_posterior,
                                                 bias_posterior_fn=l2_bias_posterior,
                                                 kernel_prior_fn=prior,
                                                 bias_prior_fn=prior,
                                                 kernel_divergence_fn=kl_div_fn,
                                                 bias_divergence_fn=kl_div_fn,
                                                 name='hidden_2')(h1)
    
    '''
    p_val = tfp.layers.DenseLocalReparameterization(1,
                                                    #bias_posterior_fn=output_bias_posterior,
                                                    kernel_prior_fn=prior,
                                                    bias_prior_fn=prior,
                                                    kernel_divergence_fn=kl_div_fn,
                                                    bias_divergence_fn=kl_div_fn,
                                                    name='p_val')(h2)

    output = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Bernoulli(logits=t, dtype=tf.float32), name='output')(p_val) 
    '''
    output = tfp.layers.DenseFlipout(2,
                                     activation='softmax',
                                     kernel_prior_fn=prior,
                                     bias_prior_fn=prior,
                                     kernel_divergence_fn=kl_div_fn,
                                     bias_divergence_fn=kl_div_fn,
                                     name='output')(h2)
    
    print('output shape:', output.shape)

    model = keras.Model(inputs=input, outputs=output, name='vi_model')

    print('*** model weights ***')
    print('with init')
    for layer in model.layers:
        if layer.name.startswith('hidden_1'):
            print(layer.weights[0])
            print('dnn weights layer 1:')
            print(l1_weights)
        elif layer.name.startswith('hidden_2'):
            print(layer.weights[0])
            print('dnn weights layer 2:')
            print(l2_weights)
    
    #expectNLL = lambda y, rv_y: -rv_y.log_prob(y)/batch_size

    model.compile(
                  loss='categorical_crossentropy',
                  #loss=expectNLL,
                  #metrics=[keras.metrics.CategoricalAccuracy()],
                  optimizer=tf.optimizers.Adam(learning_rate=params['eta'])) 
    return model
