#coding:utf-8
import glob
import os
import os.path as osp
import random
import numpy as np
import cv2
import time
import tensorflow as tf
 
from tsn_config import cfg



class Param():
  def __init__(self, data_dir, modality, num_segments, new_length, train_split_path, valid_split_path, train_batch_size, valid_batch_size, isTraining= True):
    self.data_dir= data_dir
    self.modality=modality
    self.num_segments= num_segments
    self.new_length= new_length
    self.train_split_path= train_split_path
    self.valid_split_path= valid_split_path
    self.train_batch_size= train_batch_size
    self.valid_batch_size= valid_batch_size
    self.isTraining = isTraining

param = Param(cfg.INPUT.DATA_DIR,
              cfg.INPUT.MODALITY,
              cfg.INPUT.NUM_SEGMENTS,
              cfg.INPUT.NEW_LENGTH,
              cfg.INPUT.SPLIT_PATH,
              cfg.VALID.SPLIT_PATH,
              cfg.TRAIN.BATCH_SIZE,
              cfg.VALID.BATCH_SIZE)

def set_param(data_dir, modality, valid_split_path, valid_batch_size,num_segments=cfg.INPUT.NUM_SEGMENTS, new_length=cfg.INPUT.NEW_LENGTH, train_split_path=cfg.INPUT.SPLIT_PATH,train_batch_size=cfg.TRAIN.BATCH_SIZE, isTraining=True):
  param.data_dir = data_dir
  param.modality = modality
  param.num_segments = num_segments
  param.new_length = new_length
  param.train_split_path = train_split_path
  param.valid_split_path = valid_split_path
  param.train_batch_size = train_batch_size
  param.valid_batch_size = valid_batch_size
  param.isTraining = isTraining



def _tsn_data_augment_train(sampled_images, label):
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
def _tsn_data_augment_valid(sampled_images, label):
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
  for i in range(25):
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

def _sparse_sample_valid(vid):
  vid_path= osp.join(param.data_dir, vid.split('/')[-1])# 获取路径
  if param.modality == 'rgb':
    images=[]
    rgb_frames= glob.glob(osp.join(vid_path,'img_*'))
    num_rgb_frames= len(rgb_frames)
    # 随机选25帧
    selections = range(num_rgb_frames)
    random.shuffle(selections)
    selected = selections[0:25]
    selected.sort()
    if len(selected)<25:
      for i in range(25-len(selected)):
        selected.append(selected[-1])
    for i in selected:
      image=cv2.imread(rgb_frames[i])
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
    # 随机选25帧
    selections = range(num_flow_frames)
    random.shuffle(selections)
    selected = selections[0:25]
    selected.sort()
    if len(selected)<25:
      for i in range(25-len(selected)):
        selected.append(selected[-1])
    for i in selected:
      flow_x = cv2.imread(flow_sampled[i][0], cv2.IMREAD_GRAYSCALE)
      flow_x = cv2.resize(flow_x,(340,256))/255.0
      flow_y = cv2.imread(flow_sampled[i][1], cv2.IMREAD_GRAYSCALE)
      flow_y = cv2.resize(flow_y,(340,256))/255.0
      flow = np.dstack([flow_x, flow_y])
      flows.append(flow)
    flows=np.array(flows)
    return flows

def _sparse_sample_train(vid):
  vid_path= osp.join(param.data_dir, vid.split('/')[-1])# 获取路径
  if param.modality == 'rgb':
    images=[]
    rgb_frames= glob.glob(osp.join(vid_path,'img_*'))
    num_rgb_frames= len(rgb_frames)
    average_duration= num_rgb_frames // param.num_segments
    for i in range(param.num_segments):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      snippet_begin = random.randint(begin, end-(param.new_length-1))
      snippet_end = snippet_begin+(param.new_length-1)
      rgb_sampled = rgb_frames[snippet_begin:snippet_end+1]
      if len(rgb_sampled)<param.new_length:
        for i in range(param.new_length - len(rgb_sampled)):
          rgb_sampled.append(rgb_frames[-1])
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
    average_duration= num_flow_frames // param.num_segments
    for i in range(param.num_segments):
      begin = i* average_duration
      end = begin + (average_duration - 1)
      snippet_begin = random.randint(begin, end-(param.new_length-1))
      snippet_end = snippet_begin+(param.new_length-1)
      flow_sampled = flow_frames[snippet_begin: snippet_end+1]
      if len(flow_sampled)<param.new_length:
        for i in range(param.new_length - len(flow_sampled)):
          flow_sampled.append(flow_frames[-1])
      for flow in flow_sampled:
        flow_x = cv2.imread(flow[0], cv2.IMREAD_GRAYSCALE)
        flow_x = cv2.resize(flow_x,(340,256))/255.0
        flow_y = cv2.imread(flow[1], cv2.IMREAD_GRAYSCALE)
        flow_y = cv2.resize(flow_y,(340,256))/255.0
        flow = np.dstack([flow_x, flow_y])
        flows.append(flow)
    flows=np.array(flows)
    return flows

def _generator_valid():
  vid_labels = _parse_split(param.valid_split_path) # 区别于_generator_train
  #print '@@@@@@@@@@@@@@@@@'
  #print len(vid_labels)
  for vid, label in vid_labels:
    # 稀疏采样
    sampled_images = _sparse_sample_valid(vid)
    zeros = np.zeros((cfg.NUM_CLASSES),dtype=np.int)
    zeros[int(label)-1] = 1
    label= zeros
    label=np.reshape(label,[cfg.NUM_CLASSES])
    yield (sampled_images, label)

def _generator_train():
  vid_labels = _parse_split(param.train_split_path)
  print '@@@@@@@@@@@@@@@@@'
  print len(vid_labels)
  while True:
    for vid, label in vid_labels:
      # 稀疏采样
      sampled_images = _sparse_sample_train(vid)
      zeros = np.zeros((cfg.NUM_CLASSES),dtype=np.int)
      zeros[int(label)-1] = 1
      label= zeros
      label=np.reshape(label,[cfg.NUM_CLASSES])
      yield (sampled_images, label)

def get_dataset_iter():
  """
  读取数据，预处理，组成batch，返回
  """
  #dataset=tf.data.Dataset.from_generator(generator,(tf.float32, tf.int32), (tf.TensorShape([self.num_segments*self.new_length, 224,224,3 if self.modality=='rgb' else 2]), tf.TensorShape([cfg.NUM_CLASSES]))) #()
  if param.isTraining:
    dataset_train=tf.data.Dataset.from_generator(_generator_train,(tf.float32, tf.int32) ) #()
    dataset_valid=tf.data.Dataset.from_generator(_generator_valid,(tf.float32, tf.int32) ) #()
    # 预处理map
    dataset_train=dataset_train.map(_tsn_data_augment_train,num_parallel_calls=30)
    dataset_valid=dataset_valid.map(_tsn_data_augment_valid,num_parallel_calls=4)
    # shuffle, get_batch
    #dataset_train = dataset_train.repeat().batch(param.train_batch_size).prefetch(buffer_size=10)
    dataset_train = dataset_train.repeat().shuffle(buffer_size=param.train_batch_size*20).batch(param.train_batch_size).prefetch(buffer_size=10)
    dataset_valid = dataset_valid.repeat().batch(param.valid_batch_size)
    iter_train = dataset_train.make_one_shot_iterator()
    iter_valid = dataset_valid.make_one_shot_iterator()
    return iter_train, iter_valid
  else:
    dataset_test = tf.data.Dataset.from_generator(_generator_valid,(tf.float32, tf.int32) ) #()
    # 预处理map
    dataset_test=dataset_test.map(_tsn_data_augment_valid,num_parallel_calls=4)
    # get_batch
    dataset_test = dataset_test.repeat().batch(param.valid_batch_size).prefetch(buffer_size=10)
    iter_test = dataset_test.make_one_shot_iterator()
    return iter_test
    
  '''
  image_batch, label_batch = iter.get_next()
  # 将batch reshape为batch_size*num_segments*new_length, h, w, num_channels 
  if self.modality=='rgb':
    image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 3])
  else:
    image_batch =tf.reshape(image_batch,[self.batch_size*self.num_segments*self.new_length, 224, 224, 2])
  return image_batch, label_batch
  '''
