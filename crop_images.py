from multiprocessing import shared_memory,Semaphore,Manager,Queue,Process
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
import gc
import math
import sys
import pickle

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

  shm = shared_memory.SharedMemory(name = shm_name)
  with lock:
    shm_attach_counter[shm_name] -=1

  original_organoids = np.ndarray(shape,dtype=dtype,buffer=shm.buf)

  zmin, ymin, xmin, zmax, ymax, xmax = bbox
  individual_organoid = (original_organoids[zmin:zmax, ymin:ymax, xmin:xmax]) 

  root_name = original_name.split('.')[0]
  sampleID = int(root_name.split('_')[1])

  unique_organoid_id = cantor_pair(sampleID,label) #Note: sampleID comes first -- ORDER matters in this function

  organoid_name = 'ORGANOIDS_' + f'{unique_organoid_id:03}' + '_0000.nii.gz'

  # Save cropped
  output_path = os.path.join(save_dir,organoid_name)
  sitk.WriteImage(sitk.GetImageFromArray(individual_organoid), output_path)

  shm.close()

  with lock:
      shm_counter_dict[shm_name] -= 1
      # Only unlink when both counts reach zero
      if shm_counter_dict[shm_name] == 0 and shm_attach_counter.get(shm_name, 0) == 0:
          shm.unlink()
          print('shared memory object released')
          del shm_counter_dict[shm_name]
          del shm_obj_dict[shm_name]
          del shm_attach_counter[shm_name]

    semaphore.release()


manager = Manager()
shm_counter_dict = manager.dict()
shm_obj_dict = manager.dict()
shm_attach_counter = manager.dict()
shm_lock = manager.Lock()

segmented_organoids_folder = ''
original_folder_path = ''
save_dir = ''
num_processors = ''
num_loaders = ''

names_dictionary_path = os.path.join(original_folder_path, 'original_unet_names_dictionary.pkl')

with open(names_dictionary_path,'rb') as names_dictionary_path:
  original_unet_names_dictionary = pickle.load(names_dictionary_path)


segmented_organoids_files = os.listdir(segmented_organoids_folder)
segmented_organoids_files = [file for file in segmented_organoids_files if file.endswith('.nii.gz')]

image_paths_tuples_list = []

for segmented_organoids_name in segmented_organoids_files:

 root_name = segmented_organoids_name.split('.')[0]
 imageID = int(root_name.split('_')[1])
 original_name = original_unet_names_dictionary.inv[imageID]

  segmented_organoid_path = os.path.join(segmented_organoids_folder,segmented_organoids_name)
  original_organoids_path = os.path.join(original_folder_path,original_name)

  image_paths_tuples_list.append((segmented_organoid_path,original_organoids_path,original_name))


chunk_size = math.ceil(len(image_paths_tuples_list) / num_loaders)
chunks = [image_paths_tuples_list[i * chunk_size : (i + 1) * chunk_size] for i in range(num_loaders)]

load_queue = Queue()
semaphore = Semaphore(8)
lock = Lock()

loaders = [
    Process(target=load_worker, args=(chunk, load_queue, semaphore, num_processsors,shm_counter_dict,shm_obj_dict,lock,shm_attach_counter))
    for chunk in chunks
]

processors = [
    Process(target=process_worker, args=(load_queue, semaphore,shm_counter_dict,shm_obj_dict,lock,shm_attach_counter,shm_lock))
    for _ in range(num_processsors)
]

#start all the loaders and processors

for loader in loaders:
  loader.start()

for p in processors:
  p.start()

#Wait for all loaders to finish

for loader in loaders:
  loader.join()

#when loaders finish , put None per processor to signal completion

for _ in range(num_processors):
  load_queue.put(None)

for p in processors:
  p.join()



