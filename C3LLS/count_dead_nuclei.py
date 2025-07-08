import os
import SimpleITK as sitk
from skimage.filters import threshold_otsu
import numpy as np
import json

def run_count_dead_nuclei(segmented_organoids_folder: str,
                          original_image_folder: str,
                          death_marker_channel: int,
                          json_output_path: str):


    segmented_organoids_list = [file for file in segmented_organoids_folder if file.endswith('.nii.gz')]

    for segmented_image_name in segmented_organoids_list:

        root_name = segmented_image_name.split('.')[0]
        all_channels_name = root_name + '_allChannels.nii.gz'

        segmented_organoid_path = os.path.join(segmented_organoids_folder,segmented_image_name)
        organoid_channels_path = os.path.join(original_image_folder,all_channels_name)

        segmented_organoid_cropped = sitk.ReadImage(segmented_organoid_path)
        segmented_organoid_cropped = sitk.GetArrayFromImage(segmented_organoid_cropped)

        original_image_cropped = sitk.ReadImage(organoid_channels_path)
        original_image_cropped = sitk.GetArrayFromImage(original_image_cropped)
        
        
        death_marker_channel  = original_image_cropped[:,:,:,death_marker_channel]

        flattened_image = death_marker_channel.flatten()
        threshold = threshold_otsu(flattened_image)
        death_marker_channel_thresheld = death_marker_channel > threshold
        component_labels = np.unique(segmented_organoid_cropped)
        component_labels = component_labels[component_labels > 0]  # Remove background label (0)

        components_with_overlap = []


        for component_id in component_labels:
            component_mask = (segmented_organoid_cropped == component_id)

            for z in range(segmented_organoid_cropped.shape[0]):
                component_slice = component_mask[z,:,:]
                death_channel_slice = death_marker_channel_thresheld[z,:,:]

                component_size = np.sum(component_slice > 0)
                if component_size ==0:
                    continue

            overlap = np.sum(component_slice & death_channel_slice)

            if overlap / component_size >= 0.75: # 75% of component should have overlap 

                components_with_overlap.append(component_id)
                break
                
        predicted_dead = len(components_with_overlap)
        results = {'Organoid ID': f'{segmented_image_name}',
                    'Dead Cell Count': predicted_dead} 

        with open(json_output_path,"a") as f:
                json.dump(results,f)
                f.write('\n')
                f.flush()