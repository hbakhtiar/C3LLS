import os
import SimpleITK as sitk
from multiprocessing import shared_memory,Process
import numpy as np
from skimage.measure import regionprops,label
from skimage.filters import threshold_otsu
from scipy.ndimage import center_of_mass,distance_transform_edt,binary_dilation
from skimage.morphology import disk
import pandas as pd
import gc

def cantor_pair(a,b):
      return (a+b) * (a + b + 1) //2 + b

def max_distance_centroid_to_background(component_mask):
      dist = distance_transform_edt(component_mask)
      return np.max(dist)


# Custom Process class to override the daemon flag
#I use this, but you should make sure that your server can handle
#if not, just use a regular pool with multiprocessing
class NonDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# Custom Pool class that uses NonDaemonProcess
class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super().Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc



#names got too long and confusing so renamed to binary1 and binary2
#binary 1 = segmented image of an individual organoid
#binary 2 = thresheld cell surface marker channel of the same organoid 
#function takes each nucleus (binary1) and dilates it using a structuring element = to the max distance from the nucleus' centroid to background
#if two dilated nuclei overlap, assign pixels to the closer centroid



def assign_foreground_by_boundary_expansion(binary1,binary2):
      assert binary1.shape = binary2.shape
      assigned_labels = np.zeros_like(binary1,dtype=np.int32)

      for z in range(binary1.shape[0]):
            slice1 = binary1[z]
            slice2 = binary2[z]

            if not np.any(slice2):
                  continue
      
            unique_labels = np.unique(slice1)
            unique_labels = unique_labels[unique_labels!=0] #ignore background
      
            temp_assignment = np.zeros_like(slice1,dtype=np.int32)
            label_votes = np.zeros_like(slice1, dtype=np.int32)
            label_owner = np.zeros_like(slice1, dtype=np.int32) 
      
            centroids = {}
      
            for label_val in unique_labels:
                  component_mask = slice1 == label_val
                  centroids[label_val] = np.array(center_of_mass(component_mask))
                  max_dist = max_distance_centroid_to_background(component_mask)
                  selem = disk(max_dist)
                  dilated = binary_dilation(component_mask,selem)
      
                  overlap = dilated & slice2.astype(bool)
                  overlap = dilated
      
                  label_owner[overlap] = label_val
                  label_votes[overlap] += 1
                  
                  temp_assignment[overlap] = label_val
      
            conflict_mask = label_votes > 1
            conflict_coords = np.column_stack(np.nonzero(conflict_mask))
      
            for y, x in conflict_coords:
                  distances = []
                  for label_val in unique_labels:
                        if (label_val == label_owner[y,x]):
                              centroid = centroids[label_val]
                              dist = np.linalg.norm(np.array([y,x])-centroid)
                              distances.append([dist,label_val])
                        if distances:
                              closest_label = min(distances)[1]
                              temp_assignment[y,x] = closest_label
      
            assigned_labels[z] = temp_assignment

      return assigned_labels


def sum_pixels_and_volume_by_label_fast(labeled_image,original_image):
    labels = np.unique(labeled_image)
    labels = labels[labels !=0]

    pixel_sums = ndi_sum(original_image,labels = labeled_image,index = labels)
    volumes = ndi_sum(np.ones_like(original_image),labels= labeled_image,index = labels)

    return pd.Dataframe({
        "Label": labels,
        "PixelSum": pixel_sums,
        "Volume": volumes
    })


def secondary_worker(args):

    label,bbox,cell_surface_shm_name,cell_surface_dtype,shape,nuclear_segmentation_path,sampleID,json_output_path,lock,name = args
    unique_organoid_id = cantor_pair(sampleID,label)
    unique_organoid_name = f'ORGANOID_{unique_organoid_id:03}.nii.gz'

    segmented_organoid_path = os.path.join(nuclear_segmentation_path,unique_organoid_name)

    if not os.path.exists(segmented_organoid_path):
        return

    segmented_organoid_cropped = sitk.ReadImage(segmented_organoid_path)
    segmented_organoid_cropped = sitk.GetArrayFromImage(segmented_organoid_cropped)

    #get cell surface shared object

    cell_surface_shm = shared_memory.SharedMemory(name = cell_surface_shm_name)
    cell_surface_channel = np.ndarray(shape,dtype = cell_surface_dtype,buffer = cell_surface_shm.buf)

    zmin, ymin, xmin, zmax, ymax, xmax = bbox #define bounding box 

    cell_surface_channel_cropped = (cell_surface_channel[zmin:zmax, ymin:ymax, xmin:xmax])
    flattened_image = cell_surface_channel_cropped.flatten()
    threshold = threshold_otsu(flattened_image)
    cell_surface_channel_cropped_thresheld = cell_surface_channel_cropped > threshold
    cell_surface_labeled_image = assign_foreground_by_boundary_expansion(segmented_organoid_cropped,cell_surface_channel_cropped_thresheld)
    cell_surface_scores_df = sum_pixels_and_volume_by_label(cell_surface_labeled_image,cell_surface_channel_cropped)
    results = cell_surface_scores_df.to_dict(orient='list')

    with lock:
        with open(json_output_path,"a") as f:
            json.dump(results,f)
            f.write('\n')
            f.flush()

    cell_surface_shm.close()

    del segmented_organoid_cropped,cell_surface_labeled_image,cell_surface_channel_cropped_thresheld,cell_surface_channel_cropped

    return


def primary_worker(args):

      sampleID,segmented_organoids_folder,compressed_name,nuclear_segmentation_path,json_output_path,lock,nii_gz_path,name,cell_surface_channel = args

      segmented_organoid_name = f'ORGANOID_{sampleID:03}.nii.gz'
      segmented_organoid_filepath = os.path.join(segmented_organoids_folder,segmented_organoid_name)

      segmented_organoids = sitk.ReadImage(segmented_organoid_file_path)
      segmented_organoids = sitk.GetArrayFromImage(segmented_organoids)

      segmented_organoids = skimage.measure.label(segmented_organoids,connectivity=2)
      properties = skimage.measure.regionprops(segmented_organoids)


      cell_surface_name = f'ORGANOIDS_{sampleID:03}_{cell_surface_channel:04}.nii.gz'
      cell_surface_path = os.path.join(channels_path,cell_surface_name)
      cell_surface_channel = sitk.ReadImage(cell_surface_path)
      cell_surface_channel = sitk.GetArrayFromImage(cell_surface_channel)

      #create shared memory object for the image of the channel
      shm_surface = shared_memory.SharedMemory(create = True,size = cell_surface_channel.nbytes)
      surface_shm_array = np.ndarray(cell_surface_channel.shape,dtype=cell_surface_channel.dtype,buffer=shm_surface.buf)

      args_list = [(prop.label,prop.bbox,shm_surface.name,cell_surface_channel.dtype,cell_surface_channel.shape,nuclear_segmentation_path,sampleID,json_output_path,lock,name) for prop in properties]

      with NonDaemonPool(processes=8) as pool:
            pool.map(secondary_worker,args_list)

      shm_surface.close()
      shm_surface.unlink()

      del cell_surface_channel,segmented_organoids,segmented_organoids_objects_removed
      gc.collect()

      return


segmented_organoids_folder= ''
channels_path = ''

      
      


    

args_list = [(sampleID,segmented_organoids_folder,compressed_name,nuclear_segmentation_path,json_output_path,lock,channels_path,name,cell_surface_channel) for sampleID,name in segmented_organoids_id_list]



