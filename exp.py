import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
import png
import os
from PIL.Image import fromarray
from PIL import Image
from tqdm import trange, tqdm
import cv2
import pandas as pd
import random

df = pd.read_csv("data\\train.csv").set_index("image_id")
ttdf = pd.read_csv("train_test_data.csv")
train_data = ttdf[ttdf["train"]==1].set_index("image_id")
test_data = ttdf[ttdf["train"]==0].set_index("image_id")
print(len(train_data), len(test_data), len(df))

NUM_CLASSES = 14

# def downsample(img, intercept, slope, invert=False, rescale=True):
#     img = (img*slope +intercept)
#     img_min = np.min(img)
#     img_max = np.max(img)
#     img[img<img_min] = img_min
#     img[img>img_max] = img_max
    
#     if rescale:
#         # Extra rescaling to 0-1, not in the original notebook
#         img = (((img - img_min) / (img_max - img_min))*255).astype('uint8')

#     if invert:
#         img = 255-img
    
#     return img

# def apply_clahe(ds):
#     #window_center = getattr(ds, "WindowCenter", 0)
#     #window_width = getattr(ds, "WindowWidth", 0), 
#     intercept = getattr(ds, "RescaleIntercept", 0)
#     slope = getattr(ds, "RescaleSlope", 1)
#     interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    

#     clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8,8))
#     img = downsample(ds.pixel_array, intercept, slope, interp=='MONOCHROME1')
#     return clahe.apply(img)

# def getattr(ds, attr, default):
#     try:
#         v = ds.__getattr__(attr)
#         return v
#     except AttributeError:
#         return default
   
# def fetch_dicom(filepath, image):
#     ds = dcmread(os.path.join(filepath, f"{image}.dicom"))
#     # Apply CLAHE to the image
#     img = apply_clahe(ds)
#     return img

def get_view_window(viewid, w, h):
    """
    based on the view id gets the normalized window coordinates (top left and bottom right)
    0
    1,2,3,4
    5-20
    21-84
    """
    if viewid==0:
        return [[0,w],[0,h]], 1
    elif viewid <= 4:
        viewid -= 1
        f = 2
    elif viewid <= 20:
        viewid -= 5
        f = 4
    else:
        viewid -= 21
        f = 8
    a = int(w*(viewid % f)/f)
    b = int(h*(viewid // f)/f)
    xspan = int(w/f)
    yspan = int(h/f)
    return ([[a, a+xspan], [b, b+yspan]], (1.0/f))


def imresize(im, desired_size):
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_im = Image.new('L', (desired_size, desired_size))
    new_size = tuple([int(x*ratio) for x in old_size])
    resized_im = im.resize(new_size, Image.ANTIALIAS)
    new_im.paste(resized_im, (0,0))
    return new_im, new_size

    # above is the entire image. We need to process this.
def create_masks(image, width, height):
    #me = metadata.loc[image]
    annots = df.loc[image]
    labels = set(annots['class_id'])
    masks = np.zeros(shape=[NUM_CLASSES, height, width], dtype="bool")
    for cid in labels:
        if cid==14:
            continue
        class_instances = annots[annots["class_id"]==cid]
        for index, instance in class_instances.iterrows():
            masks[cid, int(instance.y_min)-1:int(instance.y_max), int(instance.x_min)-1:int(instance.x_max)] = True 
    # convert set labels to a list
    labels = [1 if x in labels else 0 for x in range(NUM_CLASSES)]
    return masks, labels


def create_dataset(fpath, wpath, desired_size, levels, normal_proba):
    mux = sum([(2**x)**2 for x in range(levels)])
    count = 0
    outdf = pd.DataFrame(columns=['id', 'image_id', 'original_width', 'original_height', 
                                  'resized_width', 'resized_height', 'center_x', 'center_y', 'dim_ratio', 
                                  'left', 'right', 'top', 'bottom', 'c0', 'c1', 'c2', 'c3', 
                                  'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14'])
    for image in tqdm(test_data.index):
        im = Image.open(os.path.join(fpath, f"{image}.png"))
        w,h = im.size
        full_masks, labels = create_masks(image, w, h)
        for view in range(mux):
            window, dim = get_view_window(view, w, h)
            img = im.crop((window[0][0], window[1][0], window[0][1], window[1][1]))
            masks = full_masks[:, window[1][0]:window[1][1], window[0][0]:window[0][1]]
            center_x = ((window[0][0]+window[0][1])//2)/w
            center_y = ((window[1][0]+window[1][1])//2)/h
            
            labels = (np.sum(masks, axis=(1,2))>50).astype('int')
            process = True
            #print(image, view, window, labels)
            if (np.sum(labels)==0):
                # roll a die and check for probability
                # may keep complete image at least once (shall we?)
                if (view==0 and random.randint(1, 2)==1) or (view>0 and random.randint(1, int(1/normal_proba))>1):
                    process = False
            if process:
                #print(count+1, image, view, labels, np.sum(labels))
                #print(f"now saving, {image} with view: {view} as count: {count+1} with window {(window[0][0], window[1][0], window[1][0], window[1][1])}")
                im1, new_size = imresize(img, desired_size)
                count += 1
                left = window[0][0]
                top = window[1][0]
                right = window[0][1]
                bottom = window[1][1]
                newimg_name = f'{count:08d}'
                im1.save(os.path.join(wpath, f'{newimg_name}.png'))
                # test mask production
                # for cid in range(NUM_CLASSES):
                #     if labels[cid]:
                #         mi,_ = imresize(Image.fromarray(masks[cid]), desired_size)
                #         mi.save(os.path.join(wpath, f'{newimg_name}_mask_{cid}.png'))

                outdf.loc[len(outdf)] = [newimg_name, image, w, h, *new_size,  center_x, center_y, dim, 
                                         left, right, top, bottom, *labels, 1 if np.sum(labels)==0 else 0]

    outdf.to_csv("data\\testdata.csv")




fpath = 'data\\chestxrays'
wpath = 'data\\testdata'
create_dataset(fpath, wpath, 320, 3, 1/5)
# print(fpath)
# listfiles = []
# for file in os.listdir(fpath):
#     if file.endswith(".png"):
#         listfiles.append(file[:-4])
# df = pd.DataFrame(columns=["index", "image"])

# for image in tqdm(listfiles):
#     im = Image.open(os.path.join(fpath, f"{image}.png"))
#     w,h = im.size
#     im1 = im.resize((384, 384))
#     im1.save(os.path.join(wpath, f'{image}.png'))
# ds = dcmread(fpath)

# pat_name = ds.PatientName
# display_name = pat_name.family_name + ", " + pat_name.given_name
# print(f"Patient's Name...: {display_name}")
# print(f"Patient ID.......: {ds.PatientID}")
# print(f"Modality.........: {ds.Modality}")
# print(f"Study Date.......: {ds.StudyDate}")
# print(f"Image size.......: {ds.Rows} x {ds.Columns}")
# print(f"Pixel Spacing....: {ds.PixelSpacing}")

# # use .get() if not sure the item exists, and want a default value if missing
# print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

# # plot the image using matplotlib

# import pandas as pd

# df = pd.DataFrame(columns=['imageid', 'sex', 'age', 'samples_per_pixel', 'photometric_interp', 'rows', 'columns', 'pixel_spacing_x','pixel_spacing_y', 'bits_allocated', 'bits_sorted', 'high_bit', 
#                            'pixel_rep', 'pixel_aspect_ratio','window_center', 'window_width', 'rescale_intercept', 'rescale_slope', 'lossy_compression'])

# outpath = 'data\\out'
# s = np.zeros(1)
# sq = np.zeros(1)
# n = 0
# for i in trange(len(listfiles)):
#     image = listfiles[i]
#     outfile = os.path.join(outpath, f"{image}.png")
#     img = fetch_dicom(fpath, image)
#     png = Image.fromarray(img)
#     png.save(outfile)
#     x = img/255
#     s += x.sum(axis=(0,1))
#     sq += np.sum(np.square(x), axis=(0,1))
#     n += x.shape[0]*x.shape[1]

# mu = s/n
# std = np.sqrt((sq/n - np.square(mu)))
# print(mu, std, n)
# # for i in trange(len(listfiles)):
# #     ds = dcmread(listfiles[i])
    
# #     #destination = f'data\\out\\file{i:05}.png'
# #     #shape = ds.pixel_array.shape
# #     df.loc[len(df.index)] = [ listfiles[i][11:-6],
# #                              getattr(ds, 'PatientSex'),
# #                              getattr(ds, 'PatientAge'),
# #                              getattr(ds, 'SamplesPerPixel'),
# #                              getattr(ds, 'PhotometricInterpretation'),
# #                              getattr(ds, 'Rows'),
# #                              getattr(ds, 'Columns'),
# #                              getattr(ds, 'PixelSpacing'),
# #                              getattr(ds, 'BitsAllocated'),
# #                              getattr(ds, 'BitsStored'),
# #                              getattr(ds, 'HighBit'),
# #                              getattr(ds, 'PixelRepresentation'),
# #                              getattr(ds, 'PixelAspectRatio'),
# #                              getattr(ds, 'WindowCenter'),
# #                              getattr(ds, 'WindowWidth'),
# #                              getattr(ds, 'RescaleIntercept'),
# #                              getattr(ds, 'RescaleSlope'),
# #                              getattr(ds, 'LossyImageCompression'),
# #                              ]
# #     # Convert to float to avoid overflow or underflow losses.
# #     # image_2d = ds.pixel_array.astype(float)

# #     # # Rescaling grey scale between 0-255
# #     # image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

# #     # # Convert to uint
# #     # image_2d_scaled = np.uint8(image_2d_scaled)
# #     # # Write the PNG file
# #     # with open(destination, 'wb') as png_file:
# #     #     w = png.Writer(shape[1], shape[0], greyscale=True)
# #     #     size.append((shape[1], shape[0]))
# #     #     w.write(png_file, image_2d_scaled)
# #     #im = fromarray(ds.pixel_array)
# #     #plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
# #     #im.save(f"data\\out\\file{i:5}.jpg")
# # df.to_csv("data\\imagemeta.csv")
# # #plt.show()