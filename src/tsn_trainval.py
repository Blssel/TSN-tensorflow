#coding:utf-8
import argparse
import pprint
import os
import sys
import time
import tensorflow as tf
import numpy as np
import os.path as osp
import dataset_factory.data_reader as reader

from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from tsn_config import cfg,cfg_from_file ,get_output_dir
from loss import tsn_loss
from utils.view_ckpt import view_ckpt

slim = tf.contrib.slim

def average_gradients(tower_grads):
  average_grads=[]
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g,0)
      grads.append(expanded_g)
    grad = tf.concat(grads,0)
    grad = tf.reduce_mean(grad,0)
    v = grad_and_vars[0][1]
    grads_and_var = (grad, v)
    average_grads.append(grads_and_var)

  return average_grads


def _parse_args():
  parser=argparse.ArgumentParser(description='Train a keypoint regressor.')
  parser.add_argument('--cfg',dest='cfg_file',help='optional config file',default=None,type=str)
  args=parser.parse_args()
  return args

def main():

  #-------------解析参数-------------#
  args=_parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  #读取args.cfg_file文件内容并融合到cfg中
  pprint.pprint(cfg)

  #-------------任务相关配置-------------#
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
  os.environ['CUDA_VISBLE_DEVICES']=cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别

  #-------------搭建计算图-------------#
  with tf.device('/cpu:0'):
    # 操作密集型放在CPU上进行
    global_step= tf.get_variable('global_step',[],dtype=None,initializer=tf.constant_initializer(0),trainable=False)
    lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE_BASE, global_step, cfg.TRAIN.DECAY_STEP, cfg.TRAIN.DECAY_RATE, staircase=True) # 学习率
    tf.summary.scalar('learnrate',lr)
    opt = tf.train.MomentumOptimizer(lr,cfg.TRAIN.MOMENTUM)  # 优化函数
    #opt = tf.train.GradientDescentOptimizer(lr)  # 优化函数
    num_gpus = len(cfg.GPUS.split(','))
    # 建立dataset，获取iterator
    reader.set_param(cfg.INPUT.DATA_DIR,
                           cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                           cfg.VALID.SPLIT_PATH,
                           cfg.VALID.BATCH_SIZE,
                           num_segments = cfg.INPUT.NUM_SEGMENTS,
                           new_length = cfg.INPUT.NEW_LENGTH,
                           train_split_path = cfg.INPUT.SPLIT_PATH,
                           train_batch_size = cfg.TRAIN.BATCH_SIZE,
                           isTraining=True)
    ite_train, ite_valid = reader.get_dataset_iter()
    tsn_batch, label_batch = ite_train.get_next()
    tsn_batch_splits = tf.split(tsn_batch,num_or_size_splits=num_gpus, axis=0)
    label_batch_splits = tf.split(label_batch,num_or_size_splits=num_gpus, axis=0)
      
    tsn_valid_batch, label_valid_batch = ite_valid.get_next()
    
  # 在GPU上运行训练（并行）
  tower_grads = []
  with tf.variable_scope(tf.get_variable_scope()) as vscope: # 见https://github.com/tensorflow/tensorflow/issues/6220
    for i in range(num_gpus):
      with tf.device('/gpu:%d'%i), tf.name_scope('GPU_%d'%i) as scope: 
        # 获取数据,tsn_batch形式：(batch_size/num_gpus*num_seg*new_length) * h * w * num_channels
        tsn_batch_split, label_batch_split = tsn_batch_splits[i], label_batch_splits[i]
        if cfg.INPUT.MODALITY == 'rgb':
          tsn_batch_split = tf.reshape(tsn_batch_split,[cfg.TRAIN.BATCH_SIZE/num_gpus*cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, 224, 224, 3])
        elif cfg.INPUT.MODALITY == 'flow':
          tsn_batch_split = tf.reshape(tsn_batch_split,[cfg.TRAIN.BATCH_SIZE/num_gpus*cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, 224, 224, 2])
        else:
          raise ValueError("modality must be one of rgb or flow") 

        # 获取网络，并完成前传
        with slim.arg_scope(inception_v2_arg_scope()):
          logits, _= inception_v2(tsn_batch_split,
                                  num_classes=cfg.NUM_CLASSES,
                                  is_training=True,
                                  dropout_keep_prob=cfg.TRAIN.DROPOUT_KEEP_PROB,
                                  min_depth=16,
                                  depth_multiplier=1.0,
                                  prediction_fn=slim.softmax,
                                  spatial_squeeze=True,
                                  reuse=None,
                                  scope='InceptionV2',
                                  global_pool=False)
        tf.get_variable_scope().reuse_variables()
        logits = tf.reshape(logits, [cfg.TRAIN.BATCH_SIZE/num_gpus, cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, -1]) #tsn的特殊性决定
        logits = tf.reduce_mean(logits,1) # 取采样图片输出的平均值
        # 做一个batch准确度的预测
        prediction = tf.nn.softmax(logits)
        acc_batch = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(label_batch_split,1)),tf.float32))
        tf.summary.scalar('acc_on_batch',acc_batch)
        # 求loss
        for variable in tf.global_variables():
          if variable.name.find('weights')>0: # 把参数w加入集合tf.GraphKeys.WEIGHTS，方便做正则化(此句必须放在正则化之前)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS,variable)
        loss = tsn_loss(logits, label_batch_split, regularization= True)
        tf.summary.scalar('loss',loss)
        # 计算梯度，并由tower_grads收集
        grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables()) # (gradient, variable)组成的列表
        tower_grads.append(grads_and_vars)
  grads_and_vars = average_gradients(tower_grads) # 求取各GPU平均梯度
  train_step = opt.apply_gradients(grads_and_vars,global_step=global_step)
   
  # 在GPU上运行验证（串行）
  with tf.variable_scope(tf.get_variable_scope()) as vscope: # 见https://github.com/tensorflow/tensorflow/issues/6220
    with tf.device('/gpu:0'), tf.name_scope('VALID') as scope:
      tf.get_variable_scope().reuse_variables()
      if cfg.INPUT.MODALITY == 'rgb':
        tsn_valid_batch = tf.reshape(tsn_valid_batch,[cfg.VALID.BATCH_SIZE*25, 224, 224, 3])
      elif cfg.INPUT.MODALITY == 'flow':
        tsn_valid_batch = tf.reshape(tsn_valid_batch,[cfg.VALID.BATCH_SIZE*25, 224, 224, 2])
      else:
        raise ValueError("modality must be one of rgb or flow")
  
      with slim.arg_scope(inception_v2_arg_scope()):
        logits_valid, _= inception_v2(tsn_valid_batch,
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
      logits_valid = tf.reshape(logits_valid, [cfg.VALID.BATCH_SIZE, 25, -1]) #tsn的特殊性决定
      logits_valid = tf.reduce_mean(logits_valid,1) # 取采样图片输出的平均值
      # 做一个batch准确度的预测
      prediction_valid = tf.nn.softmax(logits_valid)
      acc_valid_batch = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction_valid,1),tf.argmax(label_valid_batch,1)),tf.float32))
 
   
  merged = tf.summary.merge_all()
 
  # saver
  model_variables_map={}
  for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'InceptionV2' and variable.name.find('Conv2d_1c_1x1') == -1 and variable.name.find('Momentum') == -1:  
      model_variables_map[variable.name.replace(':0', '')] = variable
  print '####################################################'
  for i in model_variables_map.keys():
    print i
  print '#####################################################'
  saver_model = tf.train.Saver(var_list=model_variables_map,max_to_keep=20) #不加载'InceptionV2/Logits/Conv2d_1c_1x1/'下的参数
  


  #-------------启动Session-------------#
  # (预测验证集，求取精度)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  config =tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
  with tf.Session(config = config) as sess:
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    joint_writer = tf.summary.FileWriter(cfg.SUMMARY_DIR, sess.graph)
    summary_writer = tf.summary.FileWriter(cfg.SUMMARY_DIR, sess.graph)

    #初始化变量(或加载pretrained models)
    tf.global_variables_initializer().run()
    saver_model.restore(sess,cfg.TRAIN.PRETRAINED_MODEL_NAME)

    sess.graph.finalize()
    start_time = time.time()
    for i in range(cfg.TRAIN.MAX_ITE):
      _,learnrate, loss_value, step, summary = sess.run([train_step, lr, loss, global_step,merged],options=run_options, run_metadata=run_metadata)
      if i==0:
        start_time = time.time()
      if i % 10 == 0:
        if i>=1:
          end_time = time.time()
          avg_time = (end_time-start_time)/float(i+1)
          print("Average time consumed per step is %0.2f secs." % avg_time)
        print("After %d training step(s), learning rate is %g, loss on training batch is %g." % (step, learnrate, loss_value))

      # 每个epoch验证一次，保存模型
      if i % 100 == 0:
        print '#############################################'
        print 'valid and save model'
        accs = []
        num = 0
        for j in range(849):  
          num+=1
          acc = sess.run(acc_valid_batch)
          accs.append(acc)
        print num
        acc_valid = np.mean(np.array(accs))
        print 'accuracy on validation set is %0.4f'%acc_valid
        print 'saving model...'
        saver_model.save(sess, cfg.TRAIN.SAVED_MODEL_PATTERN, global_step=global_step)
        print 'successfully saved !'
        print '#############################################'
        
    
      joint_writer.add_run_metadata(run_metadata, 'step%03d'%i)
      summary_writer.add_summary(summary,i)
      end_time = time.time()
      #print '%dth time step,consuming %f secs'%(i, start_time-end_time)

  summary_writer.close()

if __name__=='__main__':
  main()  
