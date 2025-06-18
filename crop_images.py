from multiprocessing import shared_memory
import numpy as np


#nnUNetv2 requries specific image IDs
# within each image, each connected component (organoid) has their own label (ID)
# use a cantor pair function to create a unique ID identifying a specific image and a specific organoid in that image
# ORDER MATTERs - a is the image ID, and b is the organoid ID - important for applying the inverse function

def cantor_pair(a,b):
  return (a+b) * (a + b + 1) //2 + b

def worker(args):
  label,bbox,shm_name, shape, dtype, save_dir, original_name = args
  shm = shared_memory.SharedMemory(name = shm_name)
  original_organoids = np.ndarray(shape,dtype=dtype,buffer=shm.buf)

  zmin, ymin, xmin, zmax, ymax, xmax = bbox
