from skimage.measure import label,regionprops
import numpy as np

def (segmented_organoid_cropped):
  num_z_layers = segmented_organoid_cropped.shape[0]

  selem_erode= disk(1) #structuring element for erosion
  selem_close = disk(3)  # Structuring element for closing

  for z in range(num_z_layers):
    prediction_layer = segmented_organoid_cropped[z,:,:]

    #erode to get rid of small pieces, restore to original size
    eroded_layer = erosion(prediction_layer,selem_erode)
    dilated_layer = dilation(eroded_layer,selem_erode)

    labeled_image = skimage.measure.label(dilated_layer)
    props = regionprops(labeled_image)

    areas = [prop.area for prop in props]
    if not areas:
      continue
    avg_area = np.mean(areas)

    min_merge_area = avg_area/3

    small_components = np.zeros_like(prediction_layer,dtype=bool)
    large_components = np.zeros_like(prediction_layer,dtype=bool)


def process_image(args,lock):

  predicted_image_name,results_folder_path,cleaned_segmentation_folder_path,x_spacing,y_spacing,z_spacing,original_study,json_output_path = args

  predicted_image_path = os.path.join(results_folder_path,predicted_image_name)
  predicted_image = sitk.ReadImage(predicted_image_path)
  predicted_image = sitk.GetArrayFromImage(predicted_image)

  predicted_image = dilate_merge_components(predicted_image)




def worker(input_queue,lock):
  while True:
    args = input_queue.get()
    if args is None:
      break
      
    process_image(args,lock)



