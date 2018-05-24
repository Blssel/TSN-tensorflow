#coding:utf-8
import tensorflow as tf
import argparse
import pprint
import os
import sys
import numpy as np
#sys.path.append("~/workspace/Classical-ActionRecognition-Model-Zoo/net")
from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from config import cfg,cfg_from_file ,get_output_dir
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
  
  #-------------搭建计算图-------------#
  # 读取数据,tsn_batch形式：(batch_size*num_seg*new_length) * h * w * num_channels
  if cfg.INPUT.MODALITY == 'rgb':
    tsn_batch ,labels= TSNDataReader(cfg.INPUT.DATA_DIR,
                                 cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                                 cfg.INPUT.NUM_SEGMENTS,
                                 cfg.INPUT.NEW_LENGTH,
                                 cfg.TRAIN.SPLIT_PATH,
                                 cfg.TRAIN.BATCH_SIZE,
                                 isTraining=False).get_batch()
  elif cfg.INPUT.MODALITY == 'flow':
    tsn_batch, labels = TSNDataReader(cfg.INPUT.DATA_DIR,
                                 cfg.INPUT.MODALITY,
                                 cfg.INPUT.NUM_SEGMENTS,
                                 cfg.INPUT.NEW_LENGTH,
                                 cfg.TRAIN.SPLIT_PATH,
                                 cfg.TRAIN.BATCH_SIZE,
                                 isTraining=False).get_batch()
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
  logits =tf.reduce_mean(logits,0)
  labels = tf.reshape(labels,[cfg.NUM_CLASSES])
  
  saver = tf.train.Saver()
  # global_step与学习率
  global_step= tf.Variable(0,name='global_step',trainable=False)
  
  # 预测验证集，求取精度 
  with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    saver.restore(sess,'models/tsn_rgb_bk_v1.ckpt-4401')
    
    sum=0
    sess.graph.finalize()
    for i in range(849):
      prediction, lab = sess.run([logits,labels])
      print prediction.shape
      print lab.shape
      prediction_ind = np.argmax(prediction,axis=0)
      #print prediction 
      print 'prediction_ind= %d'%prediction_ind
      lab_ind = np.argmax(labels,axis=0)
      #print lab
      print 'lab_ind= %d'%lab_ind
      if prediction_ind== lab_ind:
        print 'right'
        sum+=1
    accu=float(sum)/849.0

    print "the accu is %f"%accu 
if __name__=='__main__':
  main()  
  
