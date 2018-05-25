#coding:utf-8
import tensorflow as tf
slim = tf.contrib.slim


def tsn_loss(logits,labels,scope,regularization=None):
  with tf.name_scope('LossFn'):
    # 交叉熵loss 自动加入tf.GraphKeys.LOSSES中
    cross_entropy = tf.losses.softmax_cross_entropy(labels,logits)
    
    # 正则化(正则化在定义模型的时候就被加入)
    if regularization is not None:
      regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope)) # 这个scope可以保证并行时候仅加上当前GPU上的weights
      loss = cross_entropy + regularization_loss
      return loss
    
    return cross_entropy
    
