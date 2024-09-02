import tensorflow as tf
import os
from tensorflow import keras

def make_callbacks(model_dir):
    callbacks = []
    callbacks.append(_make_model_checkpoint_cb(model_dir))
    #callbacks.append(tf.keras.callbacks.LearningRateScheduler(_scheduler))
    return callbacks

def _make_model_checkpoint_cb(model_dir):
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'checkpoint{epoch:02d}-{val_loss:.2f}.h5'),
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=1
    )
    return checkpoint

''' scheduler not necessary with Adam opt
def _scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1) 
'''
