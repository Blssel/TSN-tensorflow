from tensorflow.python import pywrap_tensorflow 
def view_ckpt(checkpoint_path):
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
  var_to_shape_map = reader.get_variable_to_shape_map()  
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
  for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names 
  print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
