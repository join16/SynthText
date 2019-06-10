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
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'

def save_as_images(imname, res):
  ninstance = len(res)
  for i in range(ninstance):
    bg_img = res[i]['original']
    text_img = res[i]['img']
    x1, y1, x2, y2 = get_rough_bbox_list(res[i]['wordBB'], bg_img.shape)

    if x1 < 0 or y1 < 0:
      continue

    img = bg_img[y1:y2, x1:x2, :]
    img_text = text_img[y1:y2, x1:x2, :]

    cv2.imwrite("results/{}_{}_bg.png".format(imname, i), img)
    cv2.imwrite("results/{}_{}_text.png".format(imname, i), img_text)
    pass
  pass

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

  width = x_max - x_min
  height = y_max - y_min
  H, W, _ = shape

  if width > height:
    side = width
    y_max = y_min + side
    if y_max > H:
      diff = y_max - H
      y_min -= diff
      y_max -= diff

  if height > width:
    side = height
    x_max = x_min + side

    if x_max > W:
      diff = x_max - W
      x_min -= diff
      x_max -= diff

  return int(x_min), int(y_min), int(x_max), int(y_max)

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print(colorize(Color.RED,'Data not found and have problems downloading.',bold=True))
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


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
  db = get_data()
  print(colorize(Color.BLUE,'\t-> done',bold=True))

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print(colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
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
        save_as_images(imname, res)
        add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print(colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)