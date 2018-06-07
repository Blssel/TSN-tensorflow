#coding:utf-8
import tensorflow as tf
import argparse
import pprint
import os
import sys
import numpy as np
#sys.path.append("~/workspace/Classical-ActionRecognition-Model-Zoo/net")
from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from tsn_config import cfg,cfg_from_file ,get_output_dir
from dataset_factory import TSNDataReader

slim = tf.contrib.slim

def parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args

def main():

  #-------------解析参数-------------#
  args=parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)

  #-------------任务相关配置-------------#
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别
  os.environ['CUDA_VISBLE_DEVICES']=cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别
  
  #-------------搭建计算图-------------#
  # 读取数据,tsn_batch形式：(batch_size*num_seg*new_length) * h * w * num_channels
  with tf.device('/cpu:0'):
    # 简单计算放在CPU上进行
    ite = TSNDataReader(cfg.INPUT.DATA_DIR,
                        cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                        cfg.INPUT.NUM_SEGMENTS,
                        cfg.INPUT.NEW_LENGTH,
                        cfg.INPUT.SPLIT_PATH,
                        cfg.TRAIN.BATCH_SIZE,
                        isTraining=False).get_dataset_iter()
  # 读取数据,tsn_batch形式：(batch_size*num_seg*new_length) * h * w * num_channels
  tsn_batch, labels  = ite.get_next()
  if cfg.INPUT.MODALITY == 'rgb':
    tsn_batch = tf.reshape(tsn_batch,[cfg.TRAIN.BATCH_SIZE*cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, 224, 224, 3])
  elif cfg.INPUT.MODALITY == 'flow':
    tsn_batch = tf.reshape(tsn_batch,[cfg.TRAIN.BATCH_SIZE*cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, 224, 224, 2])
  else:
    raise ValueError("modality must be one of rgb or flow")

  # 获取网络， 并完成前传
  with slim.arg_scope(inception_v2_arg_scope()):
    logits, _= inception_v2(tsn_batch,
                            num_classes=cfg.NUM_CLASSES,
                            is_training=False,
                            dropout_keep_prob=cfg.TRAIN.DROPOUT_KEEP_PROB,
                            min_depth=16,
                            depth_multiplier=1.0,
                            prediction_fn=slim.softmax,
                            spatial_squeeze=True,
                            reuse=None,
                            scope='InceptionV2',
                            global_pool=False)
  logits = tf.reshape(logits, [cfg.TRAIN.BATCH_SIZE, cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, -1])
  logits = tf.reduce_mean(logits,1) # 取采样图片输出的平均值
  # 做一个batch准确度的预测
  prediction = tf.nn.softmax(logits)
  acc_batch = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1)),tf.float32))
  
  # saver
  model_variables_map={}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'InceptionV2' and variable.name.find('Conv2d_1c_1x1') == -1:
      model_variables_map[variable.name.replace(':0', '')] = variable
  print '####################################################'
  for i in model_variables_map.keys():
    print i
  print '#####################################################'
  saver = tf.train.Saver(var_list=model_variables_map)
  
  
  with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    saver.restore(sess,'models/tsn_rgb_bk_v1.ckpt-4401')
    
    acc = []
    sess.graph.finalize()
    for i in range(200):
      try:
       print 'testing the %dth vid'%i
       prediction = sess.run(acc_batch)
       print prediction
       acc.append(prediction)
      except:
        continue
    acc=np.mean(acc)

    print "the accu is %f"%acc 
if __name__=='__main__':
  main()  
  
