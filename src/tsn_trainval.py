#coding:utf-8
import tensorflow as tf
import argparse
import pprint
import os
import sys

#sys.path.append("~/workspace/Classical-ActionRecognition-Model-Zoo/net")
from nets.inception_v2 import inception_v2, inception_v2_arg_scope
from tsn_config import cfg,cfg_from_file ,get_output_dir
from dataset_factory import TSNDataReader
from loss import tsn_loss

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
  train_dir = get_output_dir('default' if args.cfg_file is None else args.cfg_file) # 这个设置不太好！！！！！！！！！
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
  os.environ['CUDA_VISBLE_DEVICES']=cfg.GPUS
  tf.logging.set_verbosity(tf.logging.INFO) #设置日志级别

  #-------------搭建计算图-------------#
  with tf.device('/cpu:0'):
    # 简单计算放在CPU上进行
    global_step= tf.Variable(0,name='global_step',trainable=False) # global_step
    lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE_BASE, global_step, cfg.TRAIN.DECAY_STEP, cfg.TRAIN.DECAY_RATE, staircase=True) # 学习率
    opt = tf.train.MomentumOptimizer(lr,cfg.TRAIN.MOMENTUM)  # 优化函数
    #opt = tf.train.GradientDescentOptimizer(lr)  # 优化函数
    
    num_gpus = len(cfg.GPUS.split(','))
  # 在GPU上运行（并行）
  tower_grads = []
  with tf.variable_scope(tf.get_variable_scope()) as vscope: # 见https://github.com/tensorflow/tensorflow/issues/6220
    for i in range(num_gpus):
      with tf.device('/gpu:%d'%i): 
        with tf.name_scope('GPU_%d'%i) as scope:
          # 读取数据,tsn_batch形式：(batch_size*num_seg*new_length) * h * w * num_channels
          if cfg.INPUT.MODALITY == 'rgb':
            tsn_batch ,labels= TSNDataReader(cfg.INPUT.DATA_DIR,
                                           cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                                           cfg.INPUT.NUM_SEGMENTS,
                                           cfg.INPUT.NEW_LENGTH,
                                           cfg.INPUT.SPLIT_PATH,
                                           cfg.TRAIN.BATCH_SIZE,
                                           isTraining=True).get_batch()
          elif cfg.INPUT.MODALITY == 'flow':
            tsn_batch, labels = TSNDataReader(cfg.INPUT.DATA_DIR,
                                           cfg.INPUT.MODALITY,
                                           cfg.INPUT.NUM_SEGMENTS,
                                           cfg.INPUT.NEW_LENGTH,
                                           cfg.INPUT.SPLIT_PATH,
                                           cfg.TRAIN.BATCH_SIZE,
                                           isTraining=True).get_batch()
          else:
            raise ValueError("modality must be one of rgb or flow") 

          # 获取网络，并完成前传
          with slim.arg_scope(inception_v2_arg_scope()):
            logits, _= inception_v2(tsn_batch,
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
        print '#############'
        print tf.get_variable_scope().name
        print '#############'
        logits = tf.reshape(logits, [cfg.TRAIN.BATCH_SIZE, cfg.INPUT.NUM_SEGMENTS*cfg.INPUT.NEW_LENGTH, -1]) #tsn的特殊性决定
        logits = tf.reduce_mean(logits,1) # 取采样图片输出的平均值
        prediction = tf.nn.softmax(logits)
        # 求loss
        loss = tsn_loss(logits, labels, scope, regularization= None)

        # 计算梯度，并由tower_grads收集
        grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables()) # (gradient, variable)组成的列表
        tower_grads.append(grads_and_vars)
          
  grads_and_vars = average_gradients(tower_grads) # 求取各GPU平均梯度
  train_step = opt.apply_gradients(grads_and_vars,global_step=global_step)
    
  # saver
  saver = tf.train.Saver()


  #-------------启动Session-------------#
  # (预测验证集，求取精度)
  config =tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(config = config) as sess:
    #初始化变量(或加载pretrained models)
    tf.global_variables_initializer().run()
    #saver.restore(sess,cfg.TRAIN.PRETRAINED_MODEL_NAME)

    sess.graph.finalize()
    for i in range(cfg.TRAIN.MAX_ITE):
      _,loss_value, step = sess.run([train_step,loss, global_step])

      if i % 10 == 0:
        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
      # 每个epoch验证一次，保存模型
      if i % 100 == 0:
        saver.save(sess, cfg.TRAIN.SAVED_MODEL_PATTERN, global_step=global_step) 
  
if __name__=='__main__':
  main()  
  
