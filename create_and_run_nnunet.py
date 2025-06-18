import os
import subprocess
from threading import Thread


base_directory = ''
setID = ''
setName = ''

#construct the relevant folders to be used for nnUNetV2 
# has specific requirements - reference the nnUNetV2 on the specifications


def handle_first_fold_output(process, fold):
    for line in iter(process.stdout.readline, ''):
        print(f"[Fold {fold}] {line.strip()}")
    for line in iter(process.stderr.readline, ''):
        print(f"[Fold {fold}] Error: {line.strip()}")
    process.wait()  # Wait for the first fold to complete
    print(f"First fold {fold} completed")

def construct_nnUNet_folders(base_dir,setID,setName):

  directory_list = []

  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print(f'Created base directory: {base_dir}')

  sub_directories = [f'nnUNet_raw/Dataset{setID:03}_{setName}/imagesTr',
                    f'nnUNet_raw/Dataset{setID:03}_{setName}/imagesTs',
                    f'nnUNet_raw/Dataset{setID:03}_{setName}/labelsTr']

  for sub_dir in sub_directories:
    full_path = os.path.join(base_dir,sub_dir)

    if os.path.exists(full_path):
      print(f"Directories already exist: {full_path}")
    else:
      # Create the directory if it doesn't exist
      os.makedirs(full_path)
      print(f"Created directory: {full_path}")

      directory_list.append(full_path)

    return directory_list



def wait_for_files(file_list, timeout=6000, check_interval=10):
    """Wait for the specific files to be created with a timeout."""
    start_time = time.time()
    while True:
        if all(os.path.exists(file) for file in file_list):
            print("All required files are created. Continuing with other folds.")
            return True
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for required files.")
            return False
        time.sleep(check_interval)

def extract_unique_names_with_npy(directory_path):
    """
    Extracts unique base names from files in a given directory, excluding their file extensions, 
    and appends '.npy' to each unique name.
    
    Args:
    - directory_path (str): The path to the directory containing the files.
    
    Returns:
    - unique_npy_names (set): A set of unique base names with '.npy' appended.
    """
    unique_npy_names = set()

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        # Get the base name without the extension
        base_name = os.path.splitext(filename)[0]
        # Append '.npy' to the base name
        npy_name = f"{base_name}.npy"
        # Add to the set of unique names
        npy_name = os.path.join(directory_path,npy_name)
        unique_npy_names.add(npy_name)


    return list(unique_npy_names)



#Their code is predominantly run through CLI - want to integrate it directly here
# set the command and launch as a subrocess 
# Used these defaults, but their setup is highly configurable

def call_preprocessing():

  commmand = command = [
    'nnUNetv2_plan_and_preprocess',
    '-d', str(setID),
    '-pl','nnUNetPlannerResEncL',
    '-c', '3d_fullres',
    '-np',str(16)
]

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set to whatever GPU

result = subprocess.run(command,capture_output=True,text=True)
print(result.stdout)
print(result.stderr)


def train(folds,gpu_ids,setName_and_ID,setID,trainer,configuration='3d_fullres'):

  preprocessed_folder = os.environ['nnUNet_preprocessed']
  unique_names_folder = os.path.join(preprocessed_folder,setName_and_ID,'nnUNetPlans_3d_fullres')

  required_files = extract_unique_names_with_npy(unique_names_folder)
  
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = str(first_gpu_id)  # Assign specific GPU
  env['nnUNet_n_proc_DA'] = '8'

  first_process = subprocess.Popen(
        'nnUNetv2_train',
        str(setID),
        str(configuration),
        str(first_fold),
        '-p', 'nnUNetResEncUNetLPlans',
        '-tr', str(trainer),
        '--npz' # note that i did this by default, but may want to remove if you don't have enough space 
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Start a thread to handle the output from the first fold
    output_thread = Thread(target=handle_first_fold_output, args=(first_process, first_fold))
    output_thread.start() #handle the output 






