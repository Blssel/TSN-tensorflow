#coding:utf-8
import cv2
import tensorflow as tf

def _rand_flip(image):
  """
  随机flip
  """
  return tf.image.random_flip_left_right(image)

def _crop(image, h, w, center_position=None):
  """
  center_position取值可选：None,'center','left_up','left_down','right_up','right_down'
  其作用是防止有些任务需要指定crop的位置，默认None表示随机crop
  """
    
  
def TSNDataAugment(image):
