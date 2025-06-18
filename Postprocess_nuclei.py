from skimage.measure import label,regionprops
from skimage.morphology import closing,erosion, dilation
from scipy.ndimage import binary_fill_holes
import numpy as np
from multiprocessing import Queue, Lock,Process
import os


def find_overlap(component1, component2):
    """Finds the overlap between two components represented as masks."""
    return np.sum(np.logical_and(component1, component2))

def label_and_merge_components_3d(image):
        z = image.shape[0]

        labeled_image = np.zeros_like(image,dtype=int)

        current_label = 1
        for z_layer in range(z):
                # Label connected components in the current z-layer
                layer = image[z_layer,:,:]
                labeled_layer,num_features = skimage.measure.label(layer,return_num=True)
                labeled_layer[labeled_layer==0] = -(current_label-1)
                
                # Assign the labeled components from the current layer to the 3D labeled_image
                labeled_image[z_layer,:,:] = labeled_layer + (current_label-1)
                
                # Update current_label for the next z layer (ensure unique labeling across layers)
                current_label += num_features

        #now propagate labels to the next layers
        for z_layer in range(z-1):
                current_layer = labeled_image[z_layer,:,:]
                next_layer = labeled_image[z_layer +1,:,:]

                #for each component, create a mask, apply it to the next layer, and see if any overlap
                #if there is overlap, find the 'maximally' overlapped component and assign it to the first
                #this helps identify nuclei that look like 'one' nucleus in a specific layer, but actualy split in a later layer

                for component_id in np.unique(current_layer):
                        if component_id ==0: #skip background
                                continue
                                
                        #create a mask for the component
                        component_mask = (current_layer == component_id)

                        #apply the mask to the next layer
                        next_layer_mask = (component_mask ==1) 

                        masked_next_layer = next_layer * next_layer_mask

                        #now get your overlapping components in the next layer
                        overlapping_components = np.unique(masked_next_layer[masked_next_layer > 0])
                        overlapping_components = overlapping_components[overlapping_components !=0]

                        if len(overlapping_components)==0:
                                continue
                                #if no overlap, onto the next component

                        max_overlap = 0
                        best_match_component = None
                        for overlap_component in overlapping_components:
                                #create mask for the overlapping components
                                overlap_mask = (next_layer == overlapping_component)

                                #calculate the overlap with the current component's mask
                                overlap = find_overlap(component_mask,overlap_mask)

                                #update the best match if we find a greater overlap
                                if overlap > max_overlap:
                                        max_overlap = overlap
                                        best_match_component = overlap_component

                        if best_match_component is not None:
                                next_layer[next_layer ==best_match_component] = component_id #note that we took a slice at the start so we actually updated labeled_image

        return labeled_image
        

def merge_small_with_big(segmented_organoid_cropped):
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

    for prop in props:
      if prop.area < min_merge_area:
        small_components[labeled_image == prop.label] = True
      else:
        large_components[labeled_image == prop.label] = True

    
    #close small components, let them merge with bigger components
    closed_small_components = closing(small_components, selem_close)

    #close large components, don't let them merge
    closed_large_components  = np.zeros_like(prediction_layer,dtype=int)

    large_labels = skimage.measure.label(large_components)
    current_label = 1

    large_component_masks = {}

    for region_label in range(1, large_labels.max()+ 1): #range want to include the last component hence '+1'

      #create a mask of the image with just a single large component
      # close it, ensuring it can only merge with a small component
      mask = (large_labels == region_label)

      #now close the indiviudal large component
      single_large_component = closing(mask,selem_close)

      #be sure to assign the original region_label to all connected parts
      closed_large_componets = np.where(single_large_component,region_label,closed_large_componets)

      large_component_masks[region_label] = closed_large_component
      current_label +=1

    labeled_small_components = skimage.measure.label(closed_small_components)
    final_prediction_layer = closed_large_components.copy()

    for prop in skimage.measure.regionprops(labeled_small_components):
      small_mask = (labeled_small_components == prop.label)

      #merge the small component with the big
      merged = False
      for large_label,large_mask in large_components_masks.items():
        if np.any(small_mask & large_mask):
          final_prediction_layer[small_mask] = large_label
          merged=True
          break
          
      if not merged:
        final_prediction_layer[small_mask] = current_label
        current_label +=1

    segmented_organoid_cropped[z,:,:] = binary_fill_holes(final_prediction_layer)
                
        
def process_image(args,lock):

  predicted_image_name,results_folder_path,cleaned_segmentation_folder_path,x_spacing,y_spacing,z_spacing,json_output_path = args

  predicted_image_path = os.path.join(results_folder_path,predicted_image_name)
  predicted_image = sitk.ReadImage(predicted_image_path)
  predicted_image = sitk.GetArrayFromImage(predicted_image)

  predicted_image = merge_small_with_big(predicted_image)
  predicted_image = label_merge_components_3d(predicted_image)
  organoid_cell_count = len(np.unique(predicted_image)) - (1 if 0 in predicted_image else 0)

  properties = regionprops(predicted_image)

  
    for prop in properties:
        minz, miny, minx, maxz, maxy, maxx = prop.bbox
        cropped_nucleus = predicted_image[minz:maxz, miny:maxy, minx:maxx]

        if cropped_nucleus.ndim != 3:
            continue  # or handle differently if needed

        z,y,x = cropped_nucleus.shape
        z_physical = z * z_spacing
        y_physical = y * y_spacing
        x_physical = x * x_spacing
        nuclear_volume = z_physical * y_physical * x_physical
        nuclear_volumes.append(nuclear_volume)


    cleaned_output_path = os.path.join(cleaned_segmentation_folder_path,predicted_image_name)

    z,y,x = predicted_image.shape
    z_physical = z * z_spacing
    y_physical = y * y_spacing
    x_physical = x * x_spacing

    # Physical volume
    organoid_volume = z_physical * y_physical * x_physical

    if organoid_volume >=45000:

        predicted_image = sitk.GetImageFromArray(predicted_image)
        sitk.WriteImage(predicted_image,cleaned_output_path)
        print('image written')

        results = {
            'Image ID' :predicted_image_name,
            'Organoid Count': organoid_cell_count,
            'Organoid Volume': organoid_volume,
            'Nuclear Volumes': nuclear_volumes
        }


        with lock:
            with open(json_output_path, "a") as f:
                    json.dump(results, f)
                    f.write("\n")  # Each result
                    f.flush()

        del results

    del predicted_image,nuclear_volumes,organoid_volume # can sometimes hold memory even after the process terminates, so just to be sure
    gc.collect()



def worker(input_queue,lock):
  while True:
    args = input_queue.get()
    if args is None:
      break
      
    process_image(args,lock)


def run_pipeline(args_list,num_workers=16):
    input_queue = Queue(maxsize=1000) #adjust this and num_workers based on your RAM size
    lock = Lock() #going to have multiple processes writing to the same json
    workers = []

    for _ in range(num_workers):
        p = Process(target=worker,args=(input_queue,lock_))
        p.start()
        workers.append(p)

    for args in args_list:
        input_queue.put(args)

    for _ in range(num_workers):
        input_queue.put(None)

    for p in workers:
        p.join()

x_spacing = ''
y_spacing = ''
z_spacing = ''
json_output_path = ''


segmented_organoids_directory = ''

post_processed_directory = ''

images_list = os.listdir(segmented_organoid_directory)
images_list = [image for image in images_list if image.endswith('.nii.gz')]

temp_args = [(predicted_image_name,segmented_organoids_directory,post_processed_directory, x_spacing,y_spacing,z_spacing,json_output_path) for predicted_image_name in images_list]
    
run_pipeline(args_list = temp_args,num_workers = 32)


