import os
import os.path as osp

class TSNDataReader(Object):
  def __init__(self, data_dir, split_path, modality, num_segments, new_length):
    self.data_dir= data_dir
    self.split= split
    self.modality=modality
    self.num_segments= num_segments
    self.new_length= new_length

  def _parse_split(self):
    with open(self.split,'r') as f:
      items=f.readlines()
      vid_labels=[]
      for i in items:
        item=items[i].strip().split()
        if len(item) == 3:
          vid_labels.append((item[0]+ ' '+ item[1].split('.')[0], item[-1]))
        else: # 2
          vid_labels.append((item[0].split('.')[0], item[-1]))
          
    return vid_labels
  

  def _preprocess(image):


  def _sparse_sample(self,vid):
    vid_path= osp.join(self.data_dir, vid.split('/')[-1])# 获取路径
    if self.modality== 'rgb':
      images=[] 
      rgb_frames= glob.glob('img_*')
      num_rgb_frames= len(rgbs)
      average_duration= num_rgb_frames // num_segments
      for i in range(num_segments):
        begin = i* average_duration
        end = begin + (average_duration - 1)
        snippet_begin = random.randint(begin, end-(self.new_length-1))
        snippet_end = snippet_begin+(self.new_length-1)
        rgb_sampled = rgb_frames[snippet_begin, snippet_end]
        for img in rgb_sampled:
          image=cv2.imread(img)
          # 此处可以插入预处理函数
          images.append(iamge)
      return np.array(images)
          
    else: # flow
      flows=[]
      flow_x_frames= glob.glob('flow_x_*')
      flow_y_frames= glob.glob('flow_y_*')
      assert len(flow_x_frames) == len(flow_y_frames)
      flow_frames= zip(flow_x_frames, flow_y_frames)
      num_flow_frames= len(flow_frames)
      average_duration= num_flow_frames // num_segments
      for i in range(num_segments):
        begin = i* average_duration
        end = begin + (average_duration - 1)
        snippet_begin = random.randint(begin, end-(self.new_length-1))
        snippet_end = snippet_begin+(self.new_length-1)
        flow_sampled = flow_frames[snippet_begin, snippet_end]
        for flow in flow_sampled:
          flow_x = cv2.imread(flow[0])
          # 此处可以插入预处理函数
          flow_y = cv2.imread(flow[1])
          # 此处可以插入预处理函数
          flow = np.dstack(img)
          flows.append(flow)
      return np.array(flows) 
      

  def _generator(self):
    vid_labels=_parse_split(self.split)
    for vid, label in vid_labels:
      # 稀疏采样
      sampled_images= _sparse_sample(vid)
      yield (sampled_images, label)

 
  def get_dataset():
    """
    读取数据，预处理，组成batch，返回
    """
    global _split
    _split=split
    dataset=tf.data.Dataset.from_generator(_generator,output_types=tf.float32)
    # shuffle, get_batch
  
  
        
