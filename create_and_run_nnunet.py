import os
import subprocess
from threading import Thread


base_directory = ''
setID = ''
setName = ''

#construct the relevant folders to be used for nnUNetV2 
# has specific requirements - reference the nnUNetV2 on the specifications


def handle_fold_output(process, fold):
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



#Their code is predominantly run through CLI - want to integrate it directly here
# set the command and launch as a subrocess 
# Used these defaults, but their setup is highly configurable

def call_preprocessing():

  commmand = [
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


def call_training(fold,gpu_id,setName_and_ID,setID,trainer,configuration='3d_fullres'):
  
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Assign specific GPU
  env['nnUNet_n_proc_DA'] = '8'
    

  train_process = subprocess.Popen(
        'nnUNetv2_train',
        str(setID),
        str(configuration),
        str(fold),
        '-p', 'nnUNetResEncUNetLPlans',
        '-tr', str(trainer),
        '--npz' # note that i did this by default, but may want to remove if you don't have enough space 
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Start a thread to handle the output from the first fold
    output_thread = Thread(target=handle_fold_output, args=(train_process, fold))
    output_thread.start() #handle the output 
    output_thread.join()


def call_predictions(fold,setName_and_ID,formatted_set_ID,imagesTsFolder,experiment,gpu_id,chk,trainer):

    results_path = os.path.join(os.environ['nnUNet_raw'], setName_and_ID, f'{study}_predicted_results_{chk}_{fold}_{norm}')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    command = [ 'nnUNetv2_predict',
        '-i', imagesTsFolder,
        '-o', results_path,
        '-d', str(formatted_set_ID),
        '-c', '3d_fullres',
        '-f', str(fold),
        '-p', 'nnUNetResEncUNetLPlans',
        '-tr', str(trainer),
        '-chk', f'checkpoint_{chk}.pth',
        '--disable_progress_bar'
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['DISABLE_TQDM'] = '1' #remove if you want to see the progress bar for each prediction
    env['PYTHONUNBUFFERED'] = '1'

    # Start the subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Function to handle stream reading
    def stream_reader(stream, label):
        for line in iter(stream.readline, ''):
            print(f"{label}: {line.strip()}")
        stream.close()

    # Create threads to read stdout and stderr
    # Really the simpler thing is to go into nnUNetv2 code base and stop it from printing things out
    # i handle outputs cause it can be helpful to review/monitor progress
    stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, "STDOUT"))
    stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, "STDERR"))

    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Wait for the subprocess to complete
    process.wait()

    return process.returncode




