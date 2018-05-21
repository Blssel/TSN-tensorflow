#coding:utf-8
import glob
import os
import os.path as osp
import random
import numpy as np
import cv2
import tensorflow as tf

from data_augmentation import tsn_data_augment
from config import cfg

def _parse_split(split):
  with open(split,'r') as f:
    items=f.readlines()
    videos_and_labels=[]
    for i in range(len(items)):
      item=items[i].strip().split()
      if len(item) == 3:
        videos_and_labels.append((item[0]+ ' '+ item[1].split('.')[0], item[-1]))
      elif len(item) == 2: # 2
        videos_and_labels.append((item[0].split('.')[0], item[-1]))
      else:
        raise ValueError('len of string splited is %d, check the format of video name!'%len(item))
  return videos_and_labels

def _sparse_sample(vid):
  vid_path= osp.join(cfg.INPUT.DATA_DIR, vid.split('/')[-1])# 获取路径
  print vid_path
  if cfg.INPUT.MODALITY == 'rgb':
    images=[] 
    rgb_frames= glob.glob(osp.join(vid_path,'img_*'))
    num_rgb_frames= len(rgb_frames)
    print rgb_frames
    print num_rgb_frames
    average_duration= num_rgb_frames // cfg.INPUT.NUM_SEGMENTS
    print average_duration
    for i in range(cfg.INPUT.NUM_SEGMENTS):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      print begin
      print end
      print '333333333333333333333333333333333333333333333333333'
      snippet_begin = random.randint(begin, end-(cfg.INPUT.NEW_LENGTH-1))
      snippet_end = snippet_begin+(cfg.INPUT.NEW_LENGTH-1)
      rgb_sampled = rgb_frames[snippet_begin:snippet_end+1]
      print '555555555555555555555555'
      print  rgb_sampled
      print '555555555555555555555555'
      for img in rgb_sampled:
        #image=cv2.imread(img).resize(256,340)
        image=cv2.imread(img)
        image=cv2.resize(image,(340,256))  # cv中是w*h
        image = np.reshape(image,[256,340,3])
        #image=tf.image.random_flip_left_right(image)
        print '#####################################'
        print image.shape
        print '#####################################'
        images.append(image)
    images=np.array(images)
    print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    print images.shape
    print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    images=tsn_data_augment(images,cfg.INPUT.NUM_SEGMENTS,cfg.INPUT.NEW_LENGTH)
    print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    print images.shape
    print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    return images
        
  else: # flow
    flows=[]
    flow_x_frames= glob.glob(osp.join(vid_path,'flow_x_*'))
    flow_y_frames= glob.glob('flow_y_*')
    assert len(flow_x_frames) == len(flow_y_frames)
    flow_frames= zip(flow_x_frames, flow_y_frames)
    num_flow_frames= len(flow_frames)
    average_duration= num_flow_frames // cfg.INPUT.NUM_SEGMENTS
    for i in range(cfg.INPUT.NUM_SEGMENTS):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      snippet_begin = random.randint(begin, end-(cfg.INPUT.NEW_LENGTH-1))
      snippet_end = snippet_begin+(cfg.INPUT.NEW_LENGTH-1)
      flow_sampled = flow_frames[snippet_begin: snippet_end]
      for flow in flow_sampled:
        flow_x = cv2.imread(osp.join(vid_path,flow[0]), cv2.IMREAD_GRAYSCALE)
        flow_x = cv2.resize(flow_x,(340,256))
        flow_x = np.reshape(flow_x,[256,340,2])
        flow_y = cv2.imread(osp.join(vid_path,flow[1]), cv2.IMREAD_GRAYSCALE)
        flow_y = cv2.resize(flow_y,(340,256))
        flow_y = np.reshape(flow_y,[256,340,2])
        flow = np.dstack([flow_x, flow_y])
        flows.append(flow)
    flows=np.array(flows)
    # 预处理函数（数据增强）
    flows=tsn_data_augment(flows,cfg.INPUT.NUM_SEGMENTS,cfg.INPUT.NEW_LENGTH)
    return flows

def generator():
  print '#########*****************************************************'
  vid_labels = _parse_split(cfg.TRAIN.SPLIT_PATH)
  for vid, label in vid_labels:
    # 稀疏采样
    sampled_images = _sparse_sample(vid)
    zeros = np.zeros((30),dtype=np.int)
    zeros[int(label)-1] = 1
    label= zeros
    label=np.reshape(label,[cfg.NUM_CLASSES])
    print '#########*****************************************************'
    print label.shape 
    print sampled_images
    print label
    print '#########*****************************************************'
    yield (sampled_images, label)

class TSNDataReader():
  def __init__(self, data_dir, modality, num_segments, new_length, split_path, batch_size, isTraining= True):
    self.data_dir= data_dir
    self.modality=modality
    self.num_segments= num_segments
    self.new_length= new_length
    self.split_path= split_path
    self.batch_size= batch_size
    self.isTraining = isTraining

  def get_batch(self):
    """
    读取数据，预处理，组成batch，返回
    """
    #dataset=tf.data.Dataset.from_generator(generator,(tf.float32, tf.int32), (tf.TensorShape([self.num_segments*self.new_length, 224,224,3 if self.modality=='rgb' else 2]), tf.TensorShape([cfg.NUM_CLASSES]))) #()
    dataset=tf.data.Dataset.from_generator(generator,(tf.float32, tf.int64) ) #()
    # shuffle, get_batch
    #dataset = dataset.repeat().shuffle(buffer_size=self.batch_size*50).batch(self.batch_size)
    iter = dataset.make_one_shot_iterator()
    image_batch, label_batch = iter.get_next()
    # 将batch reshape为batch_size*num_segments*new_length, h, w, num_channels 
    if self.modality=='rgb':
      image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 3])
    else:
      image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 2])
    return image_batch, label_batch



