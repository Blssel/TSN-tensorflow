#coding:utf-8
import tensorflow as tf
slim = tf.contrib.slim


def tsn_loss(logits,labels,regularization=None):
  with tf.name_scope('LossFn'):
    # 交叉熵loss 自动加入tf.GraphKeys.LOSSES中
    cross_entropy = tf.losses.softmax_cross_entropy(labels,logits)
    
    # 正则化(正则化在定义模型的时候就被加入)
    if regularization is not None:
      regularizer=tf.contrib.layers.l2_regularizer(scale=0.001)
      regularization=tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
      #regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope)) # 这个scope可以保证并行时候仅加上当前GPU上的weights
      loss = cross_entropy + regularization
      return loss
    
    return cross_entropy
    
