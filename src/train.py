#coding:utf-8
import tensorflow as tf
import argparse
import pprint

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
  # 读取数据
  if cfg.TRAIN.MODALITY == 'rgb':
    images_batch = TSNDataReader(cfg.DATA_DIR,
                                 cfg.TRAIN.SPLIT_PATH,
                                 cfg.TEST.SPLIT_PATH,
                                 cfg.TRAIN.MODALITY,
                                 cfg.TRAIN.NUM_SEGMENTS,
                                 cfg.TRAIN.NEW_LENGTH_RGB,
                                 cfg.TRAIN.BATCH_SIZE).get_dataset()
  elif cfg.TRAIN.MODALITY == 'flow':
  
  # 获取网络， 并完成前传

  #   
   
  
if __name__=='__main__':
  main()  
  
