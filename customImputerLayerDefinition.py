# -*- coding: utf-8 -*-
import tensorflow as tf

class myImputer(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.imps = self.add_weight(shape=(batch_input_shape[-1],),
                                    initializer="zeros", trainable=False)

    def call(self, X):
        return tf.where(tf.math.is_nan(X), self.imps, X)

    def adapt(self, X):
        no_nan_tf = tf.where(tf.math.is_nan(X), tf.zeros_like(X), X)
        means = tf.reduce_mean(no_nan_tf, 0)
        self.imps = tf.where(tf.math.is_nan(X), x=[means], y=X)

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
