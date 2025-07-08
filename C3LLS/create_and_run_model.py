import os
import subprocess
from threading import Thread
from nnunetv2.dataset_conversion import generate_dataset_json
import re


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
            directory_list.append(full_path)
        else:
            # Create the directory if it doesn't exist
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")

            directory_list.append(full_path)

    return directory_list



# the .json file for nnunet needs to know the max number of channels that an image can have.                         
# for now we assume that all images (training and prediction) all have the same number of channels

def get_channel_dict(directory,setName,channel=None):
    # Pattern to match the filename convention
    pattern = fr'{setName}_\d{{3}}_000(\d+)\.nii\.gz'

    if channel is not None:
        channel_names = {f'Channel {channel}': f'{channel}'}
        return channel_names


    num_channels = None
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        print(match)
        if match:
            channel_count = int(match.group(1))  # Extract the Y value
            # Update the largest Y value
            if num_channels is None or channel_count > num_channels:
                num_channels = channel_count


    channel_names = {f'Channel {i}': f'{i}' for i in range(0, num_channels + 1)}
    return channel_names

#
# need to find the number of training cases for the json file
#

def count_unique_cases(directory,setName):
    # Pattern to match the filename convention and extract cases
    pattern = fr'{setName}_0*(\d+)_000\d+'

    unique_cases = set()

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            case = int(match.group(1))  # Extract the case value
            unique_cases.add(case)

    return len(unique_cases) #just need to know the number of unique training/testing cases



def write_nnUNet_json(base_dir,setName,setID,file_ending='.nii.gz',channel=None):

    formatted_setID = f'{int(setID):03d}'
    nnUNet_directory = os.path.join(base_dir,f'nnUNet_raw/Dataset{formatted_setID}_{setName}')
    training_directory = os.path.join(nnUNet_directory,'imagesTr/')
    # Assume for now that we can get this directly from the traiing directory
    
    channel_dict = get_channel_dict(training_directory,setName=setName,channel=channel)
    num_training = count_unique_cases(training_directory,setName=setName)

    # Creating binary, so can hardcode this 
    
    labels_dict = {"background" : 0,
                   "Foreground": 1}
    
    # Hard code the file ending for now 

     


    generate_dataset_json.generate_dataset_json(output_folder =  nnUNet_directory,
                                                channel_names = channel_dict,
                                                labels=labels_dict,
                                                 num_training_cases = num_training,
                                                 file_ending = file_ending)


    return

#Their code is predominantly run through CLI - want to integrate it directly here
# set the command and launch as a subrocess 
# Used these defaults, but their setup is highly configurable




def run_preprocessing(setID: int,
                       num_processors: int):

    

    command = [
    'nnUNetv2_plan_and_preprocess',
    '-d', str(setID),
    '-pl','nnUNetPlannerResEncL',
    '-c', '3d_fullres',
    '-np',str(num_processors)
    ]

    result = subprocess.run(command,capture_output=True,text=True)
    print(result.stdout)
    print(result.stderr)



def run_training(setID, gpu_id,fold,trainer,num_processors):
      
      
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Assign specific GPU
    env['nnUNet_n_proc_DA'] = str(num_processors)

    train_process = subprocess.Popen([
        'nnUNetv2_train',
        str(setID),
        '3d_fullres',
        str(fold),
        '-p', 'nnUNetResEncUNetLPlans',
        '-tr', trainer
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Start a thread to handle the output from the first fold
    output_thread = Thread(target=handle_fold_output, args=(train_process, fold))
    output_thread.start() #handle the output 
    output_thread.join()



def run_predictions(fold,setID,gpu_id,trainer,folder2predict,results_path,chk = 'best'):

    formatted_set_ID = f'{int(setID):03d}' 

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    command = [ 'nnUNetv2_predict',
        '-i', folder2predict,
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
    stdout_thread = Thread(target=stream_reader, args=(process.stdout, "STDOUT"))
    stderr_thread = Thread(target=stream_reader, args=(process.stderr, "STDERR"))

    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Wait for the subprocess to complete
    process.wait()

    return process.returncode