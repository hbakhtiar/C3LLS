import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu, gaussian
from skimage.measure import regionprops,label
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt


#function for performing triangle thresholding
#uses threshold_triangle to get threshold, then converts image to binary
#uses SITK for image writing/reading, assumes nii.gz format
#adjust to proper image read/writer as needed
#note that assumes images are grayscaled - simply use the nuclear stain channel to grayscale

def triangle_threshold_4_binary(image):
  threshold = threshold_triangle(image)
  binary_image = (image > threshold).astype(np.uint8)
  return binary_image


def run_organoid_segmentation(filepath: str,
                              output_path: str,
                              minimum_size: int,
                              sigma: float):
    
    
    grayscale_image = sitk.ReadImage(filepath) #obtain image
    grayscale_image = sitk.GetArrayFromImage(grayscale_image) #convert to numpy array

    final_image = np.zeros_like(grayscale_image) #final_image so can populate by z layer
    num_z_layers = grayscale_image.shape[0]

    #first obtain a maximum otsu threshold value across all z layers

    max_threshold = 0

    for z in range(num_z_layers):

        grayscale_z = grayscale_image[z,:,:]
        threshold_value = threshold_otsu(grayscale_z)
        if threshold_value > max_threshold:
            max_threshold=threshold_value

#once a max threshold is found, apply across z layers

    for z in range(num_z_layers):

        grayscale_z = grayscale_image[z,:,:]
        thresheld_z = (grayscale_z > max_threshold).astype(np.uint8) 
        labeled_array = label(thresheld_z, connectivity=2)

        # Get properties of the connected components
        regions = regionprops(labeled_array)

        volume_list = []

        # Collect the areas (volume in 3D) of the connected components
        for region in regions:
            volume_list.append(region.area)

               # Compute minimum_size per z layer if user left it blank
        if minimum_size == '':
            volume_array = np.asarray(volume_list)
            z_min_size = np.mean(volume_array)
        else:
            z_min_size = float(minimum_size)  # Convert user input string to float

        labeled_array_filtered = remove_small_objects(labeled_array, min_size=z_min_size)

        # Convert the labeled array back to a binary image
        filtered_binary_image = (labeled_array_filtered > 0).astype(np.uint8)

        # Compute sigma per z layer if user left it blank
        if sigma == '':
            distance_transform = distance_transform_edt(labeled_array)
            z_sigma = np.max(distance_transform)
        else:
            z_sigma = float(sigma)  # Convert user input string to float

        blurred_z = gaussian(filtered_binary_image, sigma=z_sigma)
        segmented_layer = triangle_threshold_4_binary(blurred_z)
        final_image[z,:,:] = segmented_layer

    final_image = sitk.GetImageFromArray(final_image)
    sitk.WriteImage(final_image,output_path)

