import tensorflow as tf

def get_variables():
  global_var_list = tf.global_variables()
  trainable_var_list = tf.trainable_variables()
  
  return global_var_list, trainable_var_list
  


