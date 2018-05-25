#coding:utf-8
import glob
import os
import os.path as osp
import random
import numpy as np
import cv2
import tensorflow as tf
 
from tsn_config import cfg

def _tsn_data_augment(sampled_images, label):
  """
  接收的是未resize过的image. 256*320*channeles
  需要做crop(四个角加中央随机)  multi-scale cropping 输出224*224*channeles
  """
  size_selection=[256,224,192,168]
  position_selection=['center','left_up','left_down','right_up','right_down']
  #if cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH != sampled_images.get_shape().as_list()[0]:
  #  raise ValueError("batch size error!")
  #img_h=sampled_images.get_shape().as_list()[1]
  #img_w=sampled_images.get_shape().as_list()[2]
  img_h=256
  img_w=340
  boxes=[]
  box_ind=[]
  num=0
  # 对sampled_images，整体做一个crop
  for i in range(cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH):
    rand_h=size_selection[random.randint(0,len(size_selection)-1)] # 选择一个高宽和crop的位置
    rand_w=size_selection[random.randint(0,len(size_selection)-1)]
    position=position_selection[random.randint(0,len(position_selection)-1)]
    if position == 'center':
      boxes.append([(float(img_h)-float(rand_h))/(2.0*float(img_h)), #(img_h - rand_h)/(2*img_h)
                    (float(img_w)-float(rand_w))/(2.0*float(img_w)),
                    (float(img_h)+float(rand_h))/(2.0*float(img_h)), #((img_h-rand_h)/2+rand_h)/img_h=(img_h+rand_h)/(2*img_h)
                    (float(img_w)+float(rand_w))/(2.0*float(img_w))])
      box_ind.append(num)
      num+=1
    elif position == 'left_up':
      boxes.append([0.0,
                    0.0,
                    float(rand_h)/float(img_h),
                    float(rand_w)/float(img_w)])
      box_ind.append(num)
      num+=1
    elif position == 'left_down':
      boxes.append([(float(img_h)-float(rand_h))/float(img_h),
                     0.0,
                     1.0,
                     float(rand_w)/float(img_w)])
      box_ind.append(num)
      num+=1
    elif position == 'right_up':
      boxes.append([0.0,
                    (float(img_w)-float(rand_w))/float(img_w),
                    float(rand_h)/float(img_h),
                    1.0])
      box_ind.append(num)
      num+=1
    elif position == 'right_down':
      boxes.append([(float(img_h)-float(rand_h))/float(img_h),
                    (float(img_w)-float(rand_w))/float(img_w),
                    1.0,
                    1.0])
      box_ind.append(num)
      num+=1
  # 随机翻转
  #for i in range(0,num_segments*new_length):
  #  video_batch[i]=tf.image.random_flip_left_right(video_batch[i])
  # crop and resize
  return tf.image.crop_and_resize(sampled_images,   # (batch_size*num_segments*new_length) * img_h * img_w * num_channels
                                  boxes,
                                  box_ind,
                                  crop_size=[224,224],
                                  method='bilinear'), label

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
  if cfg.INPUT.MODALITY == 'rgb':
    images=[]
    rgb_frames= glob.glob(osp.join(vid_path,'img_*'))
    num_rgb_frames= len(rgb_frames)
    average_duration= num_rgb_frames // cfg.INPUT.NUM_SEGMENTS
    for i in range(cfg.INPUT.NUM_SEGMENTS):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      snippet_begin = random.randint(begin, end-(cfg.INPUT.NEW_LENGTH-1))
      snippet_end = snippet_begin+(cfg.INPUT.NEW_LENGTH-1)
      rgb_sampled = rgb_frames[snippet_begin:snippet_end+1]
      for img in rgb_sampled:
        image=cv2.imread(img)
        image=cv2.resize(image,(340,256))/255.0  # cv中是w*h
        images.append(image)
    images=np.array(images)
    return images

  else: # flow
    flows=[]
    flow_x_frames= glob.glob(osp.join(vid_path,'flow_x_*'))
    flow_y_frames= glob.glob(osp.join(vid_path,'flow_y_*'))
    assert len(flow_x_frames) == len(flow_y_frames)
    flow_frames= zip(flow_x_frames, flow_y_frames)
    num_flow_frames= len(flow_frames)
    average_duration= num_flow_frames // cfg.INPUT.NUM_SEGMENTS
    for i in range(cfg.INPUT.NUM_SEGMENTS):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      snippet_begin = random.randint(begin, end-(cfg.INPUT.NEW_LENGTH-1))
      snippet_end = snippet_begin+(cfg.INPUT.NEW_LENGTH-1)
      flow_sampled = flow_frames[snippet_begin: snippet_end+1]
      for flow in flow_sampled:
        flow_x = cv2.imread(osp.join(vid_path,flow[0]), cv2.IMREAD_GRAYSCALE)
        flow_x = cv2.resize(flow_x,(340,256))/255.0
        flow_y = cv2.imread(osp.join(vid_path,flow[1]), cv2.IMREAD_GRAYSCALE)
        flow_y = cv2.resize(flow_y,(340,256))/255.0
        flow = np.dstack([flow_x, flow_y])
        flows.append(flow)
    flows=np.array(flows)
    return flows

def generator():
  vid_labels = _parse_split(cfg.INPUT.SPLIT_PATH)
  for vid, label in vid_labels:
    # 稀疏采样
    sampled_images = _sparse_sample(vid)
    zeros = np.zeros((cfg.NUM_CLASSES),dtype=np.int)
    zeros[int(label)-1] = 1
    label= zeros
    label=np.reshape(label,[cfg.NUM_CLASSES])
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
    dataset=tf.data.Dataset.from_generator(generator,(tf.float32, tf.int32) ) #()
    # 预处理map
    dataset=dataset.map(_tsn_data_augment,num_parallel_calls=4) 
    # shuffle, get_batch
    if self.isTraining:
      dataset = dataset.repeat().shuffle(buffer_size=self.batch_size*10).batch(self.batch_size)
    else:
      dataset = dataset.repeat().batch(self.batch_size)
    iter = dataset.make_one_shot_iterator()
    image_batch, label_batch = iter.get_next()
    # 将batch reshape为batch_size*num_segments*new_length, h, w, num_channels 
    if self.modality=='rgb':
      image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 3])
    else:
      image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 2])
    return image_batch, label_batch
