from multiprocessing import shared_memory
import numpy as np
import SimpleITK as sitk
from multiprocessing import Semaphore
from skimage.measure import label, regionprops


#nnUNetv2 requries specific image IDs
# within each image, each connected component (organoid) has their own label (ID)
# use a cantor pair function to create a unique ID identifying a specific image and a specific organoid in that image
# ORDER MATTERs - a is the image ID, and b is the organoid ID - important for applying the inverse function

#images have the format expected from nnUNetV2
#labels should be NAME_XXX.nii.gz (XXX is imageID with leading zerois) 
#images should be split into channels with the format NAME_XXX_YYYY.nii.gz where the final 'y' tells the channel with leading zeros

def cantor_pair(a,b):
  return (a+b) * (a + b + 1) //2 + b

def load_worker(image_paths,load_queue,sempahore,num_processsors,shm_counter_dict,shm_obj_dict,lock,shm_attach_counter):

  for segmented_organoid_path, original_organoids_path,original_name in image_paths:

    semphore.acquire() #ensure that you aren't loading more than 8 images into shared memory at once

    segmented_organoids = sitk.ReadImage(segmented_organoid_path)
    segmented_organoids = sitk.GetArrayFromImage(segmented_organoids)
    segmented_organoids = skimage.measure.label(segmented_organoids,connectivity=2) # load the segmentation and get the labeled image


  
    
  
  

  


