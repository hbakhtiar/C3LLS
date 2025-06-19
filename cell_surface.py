import os
import SimpleITK as sitk
from multiprocessing import shared_memory
import numpy as np
from skimage.filters import threshold_otsu

def cantor_pair(a,b):
      return (a+b) * (a + b + 1) //2 + b

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

    
    
    


