import  tensorflow as tf

def get_earning_stop_callback():
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    return callback
