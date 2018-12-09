import matplotlib.pyplot as plt
import numpy as np
import os
from gentf import draw_object, draw_obstacles

filepath = '/Users/chingandywu/GRASP/rebuilt-dataset/re_train_dataset_600_700.txt'
num_imgs = 10
if(os.path.isfile(filepath)):
  with open(filepath) as f:
    lines = f.readlines()
    count = 0
    for line in lines:
      items = line.split(",")
      num_c_obj=int(items[1])
      obj_c=items[1:num_c_obj*3+2]
      obs_c_str=line.split(".jpg")[2]
      obs_c=obs_c_str.split(",-,")[0]
      # objstring = line.split(",-,")
      # items=objstring[1].split(",")
      # items=items[0:]
      # namech1=items[0]
      # namech2=items[1]
      img = np.ones((64,64), np.uint8)*255
      img = draw_object(img,obj_c)
      img= draw_obstacles(img,obs_c)

      plt.imshow(img)
      plt.show()
      count += 1
      if count >= num_imgs:
          break
