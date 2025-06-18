


def process_image(args,lock):

  predicted_image_name,results_folder_path,cleaned_segmentation_folder_path,x_spacing,y_spacing,z_spacing,original_study,json_output_path = args
  




def worker(input_queue,lock):
  while True:
    args = input_queue.get()
    if args is None:
      break
      
    process_image(args,lock)



