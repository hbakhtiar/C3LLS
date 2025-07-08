import os
import SimpleITK as sitk
from multiprocessing import shared_memory,Process
import numpy as np
from skimage.measure import regionprops,label
from skimage.filters import threshold_otsu
from scipy.ndimage import center_of_mass,distance_transform_edt,binary_dilation
from skimage.morphology import disk
import pandas as pd
from scipy.ndimage import sum as ndi_sum
import json

def cantor_pair(a,b):
      return (a+b) * (a + b + 1) //2 + b

def max_distance_centroid_to_background(component_mask):
      dist = distance_transform_edt(component_mask)
      return np.max(dist)


#names got too long and confusing so renamed to binary1 and binary2
#binary 1 = segmented image of an individual organoid
#binary 2 = thresheld cell surface marker channel of the same organoid 
#function takes each nucleus (binary1) and dilates it using a structuring element = to the max distance from the nucleus' centroid to background
#if two dilated nuclei overlap, assign pixels to the closer centroid



def assign_foreground_by_boundary_expansion(binary1,binary2):
      assert binary1.shape == binary2.shape
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


def run_cell_surface(segmented_organoids_folder: str,
                     original_image_folder: str,
                     cell_surface_channel: int,
                     json_output_path: str):



    segmented_organoids_list = [file for file in segmented_organoids_folder if file.endswith('.nii.gz')]

    for segmented_image_name in segmented_organoids_list:

        root_name = segmented_image_name.split('.')[0]
        all_channels_name = root_name + '_allChannels.nii.gz'

        segmented_organoid_path = os.path.join(segmented_organoids_folder,all_channels_name)
        organoid_channels_path = os.path.join(original_image_folder,all_channels_name)

        segmented_organoid_cropped = sitk.ReadImage(segmented_organoid_path)
        segmented_organoid_cropped = sitk.GetArrayFromImage(segmented_organoid_cropped)

        original_image_cropped = sitk.ReadImage(organoid_channels_path)
        original_image_cropped = sitk.GetArrayFromImage(original_image_cropped)
        
        
        cell_surface_channel_cropped = original_image_cropped[:,:,:,cell_surface_channel]

        flattened_image = cell_surface_channel_cropped.flatten()
        threshold = threshold_otsu(flattened_image)
        cell_surface_channel_cropped_thresheld = cell_surface_channel_cropped > threshold
        cell_surface_labeled_image = assign_foreground_by_boundary_expansion(segmented_organoid_cropped,cell_surface_channel_cropped_thresheld)
        cell_surface_scores_df = sum_pixels_and_volume_by_label_fast(cell_surface_labeled_image,cell_surface_channel_cropped)
        results = cell_surface_scores_df.to_dict(orient='list')


        with open(json_output_path,"a") as f:
                json.dump(results,f)
                f.write('\n')
                f.flush()