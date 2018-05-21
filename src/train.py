#coding:utf-8
import tensorflow as tf
import argparse
import pprint

from nets import inception_v2
from config import cfg,cfg_from_file ,get_output_dir
from dataset_factory import TSNDataReader

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
  train_dir = get_output_dir('default' if args.cfg_file is None else args.cfg_file) # 这个设置不太好！！！！！！！！！
  os.environ['CUDA_VISBLE_DEVICES']=cfg.GPUS
  
  #-------------搭建计算图-------------#
  # 读取数据,tsn_batch形式：(batch_size*num_seg*new_length) * h * w * num_channels
  if cfg.INPUT.MODALITY == 'rgb':
    tsn_batch ,labels= TSNDataReader(cfg.DATA_DIR,
                                 cfg.INPUT.MODALITY,  # flow模态读取方式与rgb稍有不同
                                 cfg.INPUT.NUM_SEGMENTS,
                                 cfg.INPUT.NEW_LENGTH,
                                 cfg.TRAIN.SPLIT_PATH,
                                 cfg.TRAIN.BATCH_SIZE
                                 isTraining=True).get_batch()
  elif cfg.INPUT.MODALITY == 'flow':
    tsn_batch ,labels= TSNDataReader(cfg.DATA_DIR,
                                 cfg.INPUT.MODALITY,
                                 cfg.INPUT.NUM_SEGMENTS,
                                 cfg.INPUT.NEW_LENGTH,
                                 cfg.TRAIN.SPLIT_PATH,
                                 cfg.TRAIN.BATCH_SIZE,
                                 isTraining=True).get_batch()
  else:
    raise ValueError("modality must be one of rgb or flow") 
  # 获取网络， 并完成前传
  logits = inception_v2.inception_v2(inputs,
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

  if cfg.TRAIN.MODALITY == 'rgb':
    logits = tf.resize(logits,[cfg.BATCH_SIZE,cfg.NUM_SEGMENTS*cfg.NEW_LENGTH_RGB],3) # 还原
  if cfg.TRAIN.MODALITY == 'flow':
    logits = tf.resize(logits,[cfg.BATCH_SIZE,cfg.NUM_SEGMENTS*cfg.NEW_LENGTH_FLOW],2)
  
  logits = tf.reduce_mean(logits,[2])

  # 求loss 
  loss =

  # 优化

  # 预测验证集，求取精度 
   
  
if __name__=='__main__':
  main()  
  
