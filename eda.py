from skimage import io
import pandas as pd
import numpy as np
import os
filepath = "data//temp"
size = 320
NUM_CLASSES = 14
data = pd.read_csv("data//train.csv").set_index("image_id")
meta = pd.read_csv("data//temp.csv").set_index("id")

images = meta.index
idx = 17

image = io.imread(os.path.join(filepath, f"{images[idx]:08}.png"), as_gray=True, pilmode="L")
# get the meta
info = meta.loc[images[idx]]
wr = (info.resized_width/info.original_width)/info.dim_ratio
hr = (info.resized_height/info.original_height)/info.dim_ratio
# create masks from info
#me = metadata.loc[image]
annots = data.loc[info['image_id']]
labels = [info['c0'], info['c1'], info['c2'], info['c3'], info['c4'], info['c5'], info['c6'], info['c7'], 
            info['c8'], info['c9'], info['c10'], info['c11'], info['c12'], info['c13']]
masks = np.zeros(shape=[NUM_CLASSES, size, size], dtype="float")
if not info['c14']:
    for cid in range(13):
        if labels[cid]:
            class_instances = annots[annots["class_id"]==cid]
            for index, instance in class_instances.iterrows():                        
                # get the left, bottom, top and right extents of the instance
                # in the new size. But we have not take into account our view
                
                l = max(int(instance.x_min), int(info['left'])) - int(info['left'])
                r = min(int(instance.x_max), int(info['right'])) - int(info['left'])
                b = max(int(instance.y_min), int(info['top'])) - int(info['top'])
                t = min(int(instance.y_max), int(info['bottom'])) - int(info['top'])
                print(cid, instance.x_min, instance.x_max, instance.y_min, instance.y_max, ':', l, r, b, t, ':', wr, hr, size)
                if l<r and b<t:
                    l = round(l*wr)
                    r = round(r*wr)
                    b = round(b*hr)
                    t = round(t*hr)
                    
                    masks[cid, b-1:t, l-1:r] = 1  
# convert set labels to a list
labels = list(map(int, labels))


