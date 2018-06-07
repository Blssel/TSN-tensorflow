#coding:utf-8
import tensorflow as tf
import argparse
import pprint
import os
import os.path as osp
import sys
import time

from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from tsn_config import cfg,cfg_from_file ,get_output_dir
from dataset_factory import TSNDataReader
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
    ite = TSNDataReader(cfg.INPUT.DATA_DIR,
                         cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                         cfg.INPUT.NUM_SEGMENTS,
                         cfg.INPUT.NEW_LENGTH,
                         cfg.INPUT.SPLIT_PATH,
                         cfg.TRAIN.BATCH_SIZE,
                         isTraining=True).get_dataset_iter()
    try:
      tsn_batch, label_batch = ite.get_next()
    except tf.errors.OutOfRangeError:
      pass
    tsn_batch_splits = tf.split(tsn_batch,num_or_size_splits=num_gpus, axis=0)
    label_batch_splits = tf.split(label_batch,num_or_size_splits=num_gpus, axis=0)
  # 在GPU上运行（并行）
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
        loss = tsn_loss(logits, label_batch_split, scope, regularization= None)
        tf.summary.scalar('loss',loss)
        # 计算梯度，并由tower_grads收集
        grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables()) # (gradient, variable)组成的列表
        tower_grads.append(grads_and_vars)
          
  grads_and_vars = average_gradients(tower_grads) # 求取各GPU平均梯度
  train_step = opt.apply_gradients(grads_and_vars,global_step=global_step)
   
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
  saver_model = tf.train.Saver(var_list=model_variables_map) #不加载'InceptionV2/Logits/Conv2d_1c_1x1/'下的参数
  


  #-------------启动Session-------------#
  # (预测验证集，求取精度)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
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
        saver_model.save(sess, cfg.TRAIN.SAVED_MODEL_PATTERN, global_step=global_step) 
    
      joint_writer.add_run_metadata(run_metadata, 'step%03d'%i)
      summary_writer.add_summary(summary,i)
      end_time = time.time()
      #print '%dth time step,consuming %f secs'%(i, start_time-end_time)

  summary_writer.close()

if __name__=='__main__':
  main()  
