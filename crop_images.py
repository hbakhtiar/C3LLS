from multiprocessing import shared_memory
import numpy as np
import SimpleITK as sitk
from multiprocessing import Semaphore
from skimage.measure import label, regionprops
import gc


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

    properties = measure.regionprops(segmented_organoids) #get the properties from each segment 

    shm = shared_memory.SharedMemory(create=True, size = original_organoids.nbytes)
    organoids_shm = np.ndarray(original_organoids.shape,dtype=original_organoids.dtype,buffer=shm.buffer)
    np.copyto(organoids_shm,original_organoids)

     #need to be careful about race conditions when accessing shared memory objects across multiple processes
    # establish a counter to see how many processes have attached to the object, and another to keep track of who has detached

    num_components = len(properties)
    shm_counter_dict[shm.name] = num_components
    shm_attach_counter[shm.name] = num_components
    shm_obj_dict[shm.name] = True

    #now load your properties to the queue to be processed

    for prop in properties:
      load_queue.put((
                prop.label,
                prop.bbox,
                shm.name,
                original_organoids.shape,
                original_organoids.dtype,
                save_dir,
                original_name
            ))

      del original_organoids, segmented_organoids #objects have been loaded to shared memory. delete now so they don't hold onto the process' memory
      gc.collect()


def process_worker(load_queue,semaphore,shm_counter_dict,shm_obj_dict,lock,shm_attach_counter,shm_lock):

  while True:
    item = load_queue.get()

  if item is None:
    break

  label, bbox, shm_name, shape, dtype, save_dir,original_name = item

    
  
  

  


