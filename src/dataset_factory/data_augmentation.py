#coding:utf-8
import cv2
import random
import tensorflow as tf

""" 
def tsn_data_augment(video_batch,batch_size,num_segments,new_length):
  size_selection={256,224,192,168}
  position_selection={'center','left_up','left_down','right_up','right_down'}
 
  if batch_size != len(video_batch):
    raise ValueError("batch size error!")
  img_h=video_batch.get_shape.as_list()[2]
  img_w=video_batch.get_shape.as_list()[3]
  boxes=[]
  box_ind=[]
  num=0
  # 对于每一个video，做一个crop
  for i in range(batch_size):
    rand_h=size_selection[random.randint(0,len(size_selection)-1)] # 选择一个高宽和crop的位置
    rand_w=size_selection[random.randint(0,len(size_selection)-1)]
    position=position_selection[random.randint(0,len(position_selection)-1)]
    num_images=len(video_batch[i])
    for j in range(num_images):
      if position == 'center':
        boxes.append([(float(img_h)-float(rand_h))/(2.0*float(img_h)), #(img_h - rand_h)/(2*img_h)
                      (float(img_w)-float(rand_w))/(2.0*float(img_w)),
                      (float(img_h)+float(rand_h))/(2.0*float(img_h)), #((img_h-rand_h)/2+rand_h)/img_h=(img_h+rand_h)/(2*img_h)
                      (float(img_w)+float(rand_w))/(2.0*float(img_w))])
        box_ind.append(num++)
      elif position == 'left_up':
        boxes.append([0.0,
                      0.0,
                      float(rand_h)/float(img_h),
                      float(rand_w)/float(img_w)])
        box_ind.append(num++)
      elif position == 'left_down':
        boxes.append([(float(img_h)-float(rand_h))/float(img_h),
                       0.0,
                       1.0,
                       float(rand_w)/float(img_w)])
        box_ind.append(num++)
      elif position == 'right_up':
        boxes.append([0.0,
                      (float(img_w)-float(rand_w))/float(img_w),
                      float(rand_h)/float(img_h),
                      1.0])
        box_ind.append(num++)
      elif position == 'right_down':
        boxes.append([(float(img_h)-float(rand_h))/float(img_h),
                     (float(img_w)-float(rand_w))/float(img_w),
                      1.0, 
                      1.0])
        box_ind.append(num++)

  # 随机翻转
  for i in range(0,len(batch_size)):
    video_batch[i]=tf.image.random_flip_left_right(video_batch[i])
  # reshpe 一下
  video_batch=tf.reshape(video_batch,[batch_size*num_segments*new_length,img_h,img_w,-1])
  # crop and resize
  return tf.image.crop_and_resize(video_batch,   # (batch_size*num_segments*new_length) * img_h * img_w * num_channels
                                  boxes,
                                  box_ind,
                                  crop_size=[224,224],
                                  method='bilinear') 

"""







def tsn_data_augment(video_batch,num_segments,new_length):
  """
  接收的是未resize过的image
  """
  size_selection=[256,224,192,168]
  position_selection=['center','left_up','left_down','right_up','right_down']
  if num_segments*new_length != len(video_batch):
    raise ValueError("batch size error!")
  img_h=video_batch.shape[1]
  img_w=video_batch.shape[2]
  boxes=[]
  box_ind=[]
  num=0
  # 对video，整体做一个crop
  for i in range(num_segments*new_length):
    rand_h=size_selection[random.randint(0,len(size_selection)-1)] # 选择一个高宽和crop的位置
    rand_w=size_selection[random.randint(0,len(size_selection)-1)]
    position=position_selection[random.randint(0,len(position_selection)-1)]
    print '444444444444444444444444' 
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
    
  print '555555555555555555555555555'
  # 随机翻转
  #for i in range(0,num_segments*new_length):
  #  video_batch[i]=tf.image.random_flip_left_right(video_batch[i])
  print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
  # crop and resize
  return tf.image.crop_and_resize(video_batch,   # (batch_size*num_segments*new_length) * img_h * img_w * num_channels
                                  boxes,
                                  box_ind,
                                  crop_size=[224,224],
                                  method='bilinear') 





















