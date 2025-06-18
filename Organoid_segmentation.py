
import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects



#function for performing triangle thresholding
#uses threshold_triangle to get threshold, then converts image to binary

def triangle_threshold_4_binary(image):
  threshold = threshold_triangle(image)
  binary_image = (image > threshold).astype(np.uint8)
  return binary_image

#Images should be converted to .nii.gz format, with only nuclear stain channel
#Image should have format (z,y,x)

images_folder = ''
saveImage = ''
nuclear_stain = ''
minimum_size = ''

images_2_segment_list = os.listdir(images_folder)
images_2_segment_list = [image_name for image_name in images_2_segment_list if image_name.endswith('.nii.gz')]

if images_2_segment_list is None:
  raise ValueError("Image folder empty or files aren't in .nii.gz format")


for image_name in image_2_segment_list:

    image_path = os.path.join(image_folder,image_name)

    grayscale_image = sitk.ReadImage(image_path) #obtain image
    grayscale_image = sitk.GetArrayFromImage(grayscale_image) #convert to numpy array
    thresheld_image = triangle_threshold_4_binary(grayscale_image) #threshold image
  
    final_image = np.zeros_like(grayscale_image) #final_image so can populate by z layer
    num_z_layers = grayscale_image.shape[0]

    #first obtain a maximum otsu threshold value across all z layers
  
    max_threshold = 0

    for z in range(num_z_layers):

        grayscale_z = grayscale_image[z,:,:]
        threshold_value = filters.threshold_otsu(grayscale_z)
        if threshold_value > max_threshold:
            max_threshold=threshold_value

  #once a max threshold is found, apply across z layers
  
    for z in range(num_z_layers):

        grayscale_z = grayscale_image[z,:,:]
        thresheld_z = (grayscale_z > max_threshold).astype(np.uint8) 
        labeled_array = skimage.measure.label(thresheld_z, connectivity=2)

        # Get properties of the connected components
        regions = regionprops(labeled_array)

        volume_list = []

        # Collect the areas (volume in 3D) of the connected components
        for region in regions:
            volume_list.append(region.area)

        if min_size is None:
          
          # Calculate the average volume of the regions, use average volume by default
          volume_array = np.asarray(volume_list)
          min_size = np.mean(volume_array)

        # Remove regions smaller than the average volume
        labeled_array_filtered = remove_small_objects(labeled_array, min_size=min_size) #
        # labeled_array_filtered = labeled_array_filtered

        # Convert the labeled array back to a binary image
        filtered_binary_image = (labeled_array_filtered > 0).astype(np.uint8)

        # 2. Perform the distance transform on the binary image
        distance_transform = distance_transform_edt(labeled_array)


        max_distance = np.max(distance_transform)

        blurred_z = skimage.filters.gaussian(filtered_binary_image, sigma=max_distance/3)
        segmented_layer = loadFunctions.triangle_threshold(blurred_z)
        final_image[z,:,:] = segmented_layer

