
import os


#Images should be converted to .nii.gz format before use
#

images_folder = ''
saveImage = ''
nuclear_stain = ''

images_2_segment_list = os.listdir(images_folder)
images_2_segment_list = [image_name for image_name in images_2_segment_list if image_name.endswith('.nii.gz')]

if images_2_segment_list is None:
  raise ValueError("Image folder empty or files aren't in .nii.gz format")


for image in image_2_segment_list:

    image_path = os.path.join(image_folder,image)

    original_image = sitk.ReadImage(image_path)
    grayscale_image = sitk.GetArrayFromImage(original_image)

    segmented_cells_list=[]
    original_names = []


    #convert back to nparray
    max_threshold = 0

    final_image = np.zeros_like(grayscale_image)
    num_z_layers = grayscale_image.shape[0]

        #convert back to nparray

    thresheld_image = loadFunctions.triangle_threshold(grayscale_image)

    num_z_layers = thresheld_image.shape[0]
    final_image = np.zeros_like(thresheld_image)
    filtered_image = np.zeros_like(thresheld_image)
    max_threshold = 0


    for z in range(num_z_layers):


        grayscale_z = grayscale_image[z,:,:]
        threshold_value = filters.threshold_otsu(grayscale_z)
        if threshold_value > max_threshold:
            max_threshold=threshold_value




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

        # Calculate the average volume of the regions
        volume_array = np.asarray(volume_list)
        average_volume = np.mean(volume_array)
        standard_dev = np.std(volume_array)

        # Remove regions smaller than the average volume
        labeled_array_filtered = remove_small_objects(labeled_array, min_size=average_volume)
        # labeled_array_filtered = labeled_array_filtered

        # Convert the labeled array back to a binary image
        filtered_binary_image = (labeled_array_filtered > 0).astype(np.uint8)

        # 2. Perform the distance transform on the binary image
        distance_transform = distance_transform_edt(labeled_array)


        max_distance = np.max(distance_transform)

        blurred_z = skimage.filters.gaussian(filtered_binary_image, sigma=max_distance/3)
        segmented_layer = loadFunctions.triangle_threshold(blurred_z)
        final_image[z,:,:] = segmented_layer

