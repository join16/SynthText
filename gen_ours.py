# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import _pickle as cp
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 10 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'

OUR_DATA_PATH = "data/bg_preproc"
IMG_DIR = osp.join(OUR_DATA_PATH, "bg_img")

def save_as_images(img_index, res):
  ninstance = len(res)
  for i in range(ninstance):
    bg_img = res[i]['original']
    text_img = res[i]['img']
    x1, y1, x2, y2 = get_rough_bbox_list(res[i]['wordBB'], bg_img.shape)

    if x1 < 0 or y1 < 0:
      continue

    img = bg_img[y1:y2, x1:x2, :]
    img_text = text_img[y1:y2, x1:x2, :]

    cv2.imwrite("results/synth/bg/{}_{}.png".format(img_index, i), img)
    cv2.imwrite("results/synth/text/{}_{}.png".format(img_index, i), img_text)
    pass
  pass

SIZE = 256
def get_rough_bbox_list(bbox_list, shape):
  bbox_list = np.array(bbox_list)
  x_coords = np.array([])
  y_coords = np.array([])

  for word_idx in range(bbox_list.shape[-1]):
    bbox = bbox_list[:, :, word_idx]
    x_coords = np.concatenate([x_coords, bbox[0]])
    y_coords = np.concatenate([y_coords, bbox[1]])

  x_min = np.min(x_coords)
  x_max = np.max(x_coords)
  y_min = np.min(y_coords)
  y_max = np.max(y_coords)

  x_mid = int(np.mean([x_min, x_max]))
  y_mid = int(np.mean([y_min, y_max]))

  H, W, _ = shape
  half_size = SIZE // 2

  x_start = x_mid - half_size
  x_end = x_mid + half_size
  y_start = y_mid - half_size
  y_end = y_mid + half_size

  if x_start < 0:
    x_end -= x_start
    x_start = 0
  elif x_end > W:
    diff = x_end - W
    x_end = W
    x_start -= diff

  if y_start < 0:
    y_end -= y_start
    y_start = 0
  elif y_end > H:
    diff = y_end - H
    y_end = W
    y_start -= diff

  return int(x_start), int(y_start), int(x_end), int(y_end)

def get_data():
  seg_db = h5py.File(osp.join(OUR_DATA_PATH, "seg.h5"))
  depth_db = h5py.File(osp.join(OUR_DATA_PATH, "depth.h5"))

  return {
    "seg": seg_db["mask"],
    "depth": depth_db
  }

def get_filtered_imnames():
  with open(osp.join(OUR_DATA_PATH, 'imnames.cp'), 'rb') as f:
    return list(cp.load(f))


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    db['data'][dname].attrs['txt'] = res[i]['txt']

def main(viz=False):
  # open databases:
  print(colorize(Color.BLUE,'getting data..',bold=True))
  print(colorize(Color.BLUE,'\t-> done',bold=True))

  # get the names of the image files in the dataset:
  imnames = get_filtered_imnames()
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  db = get_data()

  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image
      img_path = osp.join(IMG_DIR, imname)
      if not osp.exists(img_path):
        continue
      img = Image.open(img_path)
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:].T
      depth = depth[:,:,1]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print(colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        save_as_images(i, res)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print(colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)