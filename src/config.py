#coding:utf-8
import os
import os.path as osp
from easydict import EasyDict as edict

"""cfg变量中保存的是默认config
ymal文件保存变动性config
"""
__C = edict()
cfg = __C  # 引用传递

#------关于输入的配置-------#
__C.INPUT = edict()

__C.INPUT.INPUT_IMAGE_FORMAT = 'normal'

#------Training配置-------#
__C.TRAIN = edict()

__C.TRAIN.SPLIT_PATH = 'data/..'
__C.TRAIN.MODALITY= 'rgb'
__C.TRAIN.NUM_SEGMENTS_RGB = 3
__C.TRAIN.NEW_LENGTH_RGB = 1
__C.TRAIN.NEW_LENGTH_FLOW = 10

__C.TRAIN.BATCH_SIZE = 10

#------Test配置-------#
__C.TEST = edict()

__C.TEST.SPLIT_PATH = 'data/..'

__C.TEST.BATCH_SIZE = 10

#------Network配置-------#
__C.NET = edict()
__C.NET.TRAIN_TOP_BN = False
__C.NET.DROPOUT = -1.0

#------其它配置-------#
__C.GPUS = '0'
__C.EXP_DIR = 'expt_outputs/' #输出文件夹设置在项目目录下expt_outputs文件夹中
__C.DATA_DIR = '/share/dataset/..'
__C.RGB_PREFIX = 'img_'
__C.FLOW_X_PREFIX = 'flow_x'
__C.FLOW_Y_PREFIX = 'flow_y'
__C.split



def get_output_dir(config_file_name):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.EXP_DIR, osp.basename(config_file_name)))
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.iteritems():
    # a must specify keys that are in b
    if not b.has_key(k):
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                         'for config key: {}').format(type(b[k]),
                                                      type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)
