import sys
import shutil
import os
from .Organoid_segmentation import run_organoid_segmentation
from .organoid_cell_segmentation import run_nuclear_segmentation
from .single_cell_segmentation import run_single_cell_segmentation
from .create_and_run_model import run_preprocessing, run_training,run_predictions, construct_nnUNet_folders,write_nnUNet_json
from .count_dead_nuclei import run_count_dead_nuclei
from .cell_surface import run_cell_surface


def main_menu():
    while True:
        print("\n--- C3LLS Main Menu ---")
        print("\n--- Refer to our github page github.com/hbakhtiar/C3LLS for instructions ---")
        print("1. Create Training Data")
        print("2. Make a Model (needs training data)")
        print('3. Predict from existing model')
        print("4. Count Dead Nuclei (needs a trained nuclear model)")
        print("5. Compute Cell Surface Score (needs a trained nuclear model)")
        print('6: Exit')

        choice = input("Enter your choice (1-6): ")     

        if choice == '1':
            auto_seg_menu()
            
        elif choice =='2':
            nnUNet_menu()

        elif choice =='3':
            prediction_menu()

        elif choice =='4':
            dead_nuclei_menu()
            
        elif choice =='5':
            cell_surface_menu()

        elif choice =='6':
            print('Exiting!')
            sys.exit()
        
        else:
            print('Invalid input. Only numerical options allowed.')

def auto_seg_menu():
    while True:
        print('\n --- What type of training data would you like to create? ---')
        print('1. Training data for finding organoids')
        print('2. Training data for finding nuclei within an organoid')
        print('3. Training data for finding single cells (not in an organoid)')
        print('4. Return to main menu')

        choice = input('Enter your choice (1-4): ')

        if choice == '1':
            organoid_seg_menu()

        elif choice =='2':
            nuclear_seg_menu()

        elif choice =='3':
            single_cell_seg_menu()
        elif choice == '4':
            print('Returning to main menu')
            break
        else:
            print("Invalid Input. Only numerical options allowed.")


def organoid_seg_menu():

    while True:
        print('\n --- Organoid training data creation ---')
        print('1. Auto seg an individual image?')
        print('2. Auto seg multiple images?')
        print('3. Return to previous menu')

        choice = input('Enter your choice (1-3): ')

        if choice =='1':

            while True:

                filepath  = input('Enter image file path: ')
                output_path = input('Enter output path for segmented image: ')
                minimum_size = input('Enter minimum organoid size to keep (integer): ')
                sigma = input('Enter sigma (leave blank for default): ')

                run_organoid_segmentation(filepath,output_path,minimum_size,sigma)

                segment_more = input('Would you like to segment another image? (Y/N): ')

                if segment_more.strip().upper() != 'Y':
                    break

        elif choice == '2':
            folder_path = input('Enter images folder path: ')
        
        elif choice =='3':
            print('Returning to auto seg menu')
            break
        else:
            print("Invalid Input. Only numerical options allowed.")


def nuclear_seg_menu():
    while True:
            print('\n --- Create Nuclear Training Data ---')
            print('1. Auto Seg an individual image')
            print('2. Auto Seg multiple images at once')
            print('3. Return to main menu')

            choice = input('Enter your choice (1-3): ')

            if choice == '1':

                while True: 
                    print('\n --- Reference github section Nuclear Segmentation for parameter descriptions ---')
                    filepath = input('Enter image file path: ')
                    output_path = input('Enter output path for segmented image: ')
                    percentile = input('Enter threhsold percentile (leave blank for default): ')
                    max_freq = input('Enter max frequency (leave blank for default): ')
                    frequency_step = input('Enter frequency step (leave blank for default): ')
                    sigma = input('Enter log-gabor sigma (leave blank for default): ')
                    remove_background = input('Attempt to remove background noise (leave blank for default)? ')

                    run_nuclear_segmentation(filepath,
                                            output_path,
                                            percentile,
                                            max_freq,
                                            frequency_step,
                                            sigma,
                                            remove_background)
                    
                    segment_more = input('Would you like to segment another image? (Y/N): ')

                    if segment_more.strip().upper() != 'Y':
                        break


            elif choice =='2':
                folder_path = input('Enter folder path: ')
                    
            elif choice =='3':
                print('Returning to auto seg menu')
                break
            else:
                print("Invalid Input. Only numerical options allowed.")



def single_cell_seg_menu():
    while True:
            print('\n --- Create Single Cell Training Data ---')
            print('1. Auto Seg an individual image')
            print('2. Auto Seg multiple images at once')
            print('3. Return to main menu')

            choice = input('Enter your choice (1-3): ')


            if choice =='1':

                while True:

                    filepath  = input('Enter image file path: ')
                    output_path = input('Enter output path for segmented image: ')
                    minimum_size = input('Enter minimum organoid size to keep (integer): ')
                    percentile = input('Enter percentile threshold: ')

                    run_single_cell_segmentation(filepath,output_path,minimum_size,percentile)

                    segment_more = input('Would you like to segment another image? (Y/N): ')

                    if segment_more.strip().upper() != 'Y':
                        break   

            elif choice =='2':
                folder_path = input('Enter folder path: ')
                    
            elif choice =='3':
                print('Returning to auto seg menu')
                break
            else:
                print("Invalid Input. Only numerical options allowed.")       


def nnUNet_menu():


    while True:
        print('\n --- Create C3LLS Model ---')
        print('\n --- See GitHub page for parameter instructions ---')

        print('1. Create New Model')
        print('2. Train previously preprocessed model')
        print('3. Return to main menu')

        choice = input('Enter you choice (1-3): ')

        if choice == '1':
            base_directory = input('Enter base directory for model: ')
            setID = int(input('Enter set ID (int): '))
            setName = input('Enter set name (string): ')
            num_processors = int(input('Enter number of processors to use (int): '))
            channel2segment = input('Enter channel for segmentation: ')

            preprocess = input('Continue to preprocessing (Y/N): ')
            if preprocess.strip().upper() != 'Y':
                
                continue  # Restarts the function from the top of the loop

            # If user proceeds with preprocessing, break out of loop and continue

            print('Constructing Folders for Model...')
            directory_list = construct_nnUNet_folders(base_dir=base_directory,
                                     setID=setID,
                                     setName=setName)
            
            training_folder = input('Enter folder with training images (leave blank if imagesTr full): ')
            testing_folder = input('Enter folder with validation images (leave blank if imagesTs full): ')
            labels_folder = input('Enter folder with training labels (leave blank if labelsTr full): ')

            if os.path.exists(training_folder):
                shutil.copytree(training_folder,directory_list[0],dirs_exist_ok=True)
            
            if os.path.exists(testing_folder):
                shutil.copytree(testing_folder,directory_list[1],dirs_exist_ok=True)

            if os.path.exists(labels_folder):
                shutil.copytree(labels_folder,directory_list[2],dirs_exist_ok=True)
            

            print('Preprocessing...')

            #Set environment variables
            os.environ['nnUNet_raw'] = os.path.join(base_directory,'nnUNet_raw')
            os.environ['nnUNet_preprocessed'] = os.path.join(base_directory,'nnUNet_preprocessed')
            os.environ['nnUNet_results'] = os.path.join(base_directory,'nnUNet_results')

            write_nnUNet_json(base_directory,setName,setID,file_ending='.nii.gz',channel=channel2segment)
            run_preprocessing(setID=setID,num_processors=num_processors)


        elif choice =='2':
            setID = int(input('Enter set ID (int): '))
            gpu_id = input('Enter GPU ID: ')
            fold = input('Enter fold to train: ')
            num_processors = input('Enter number of processors: ')
            base_directory = input('Enter base directory for model: ')

            #Set environment variables
            os.environ['nnUNet_raw'] = os.path.join(base_directory,'nnUNet_raw')
            os.environ['nnUNet_preprocessed'] = os.path.join(base_directory,'nnUNet_preprocessed')
            os.environ['nnUNet_results'] = os.path.join(base_directory,'nnUNet_results')


            while True:

                print('\n What is this model for? (determines trainer)')
                print('1. Organoid Segementation (finding organoids)')
                print('2. Nuclear Segmentation within an organoid')
                print('3. Single cell segmentation (not in an organoid)')

                choice = input('Enter choice (1-3): ')

                if choice =='1' or choice =='3':
                    trainer = 'nnUNetTrainerCosAnneal'
                    break

                elif choice =='2':
                    trainer = 'nnUNetTrainerCELoss'
                    break

                else: 
                    print("Invalid Input. Only numerical options allowed.")       

            train_model = input('Continue to model training (Y/N): ')
            if train_model.strip().upper() !='Y':
                continue

            print('Training...')

            run_training(setID,gpu_id,fold,trainer,num_processors)


        elif choice =='3':
            print('Returning to main menu')
            break

    
def prediction_menu():

    print('\n --- Run C3LLS predictions ---')
    print('\n --- See GitHub page for parameter instructions ---')

    print('1. Predict Existing Model')
    print('2. Return to main menu')

    choice = input("Enter choice (1-2): ")

    if choice =='1':

        folder2predict = input('Enter folder with images to predict: ')
        fold = input('Fold for predictions (int): ')
        setID = input('Set ID to predict: ')
        results_path = input('Results output path: ')
        gpu_id = input('Enter GPU ID: ')

        #perform some type of input validation

        while True:

            print('\n What is this model for? (determines trainer)')
            print('1. Organoid Segementation (finding organoids)')
            print('2. Nuclear Segmentation within an organoid')
            print('3. Single cell segmentation (not in an organoid)')

            choice = input('Enter choice (1-3): ')

            if choice =='1' or choice =='3':
                trainer = 'nnUNetTrainerCosAnneal'
                break

            elif choice =='2':
                trainer = 'nnUNetTrainerCELoss'
                break

            else: 
                print("Invalid Input. Only numerical options allowed.")    

        run_predictions(fold,setID,gpu_id,trainer,folder2predict,results_path)


    elif choice =='2':
        print('Returning to main menu')
        return

def dead_nuclei_menu():

    while True:

        print('\n --- Count Dead Nuclei ---')
        print('\n --- See Github for description ---')

        print('1. Count dead nuclei with existing model')
        print('2. Return to main menu')

        choice = input('Enter choice (1-2): ')

        if choice =='1':

            segmented_organoids_folder = input('Enter folder with segmented organoids: ')
            original_image_folder = input('Enter folder with original images: ')
            death_marker_channel = input('Enter death marker channel (int;): ')
            json_output_path = input('Enter output path for json file: ')

            #perform some type of input validation

            run_count_dead_nuclei(segmented_organoids_folder,
                            original_image_folder,
                            death_marker_channel,
                            json_output_path)
            
        elif choice =='2':
            print('Returning to main menu')
            break
        
        else:
            print("Invalid Input. Only numerical options allowed.")      

def cell_surface_menu():
    
    while True:

        print('\n --- Quantify Cell Surface Marker ---')
        print('\n --- See Github for description ---')

        print('1. Quantify surface marker with exisiting model')
        print('2. Return to main menu')

        choice = input('Enter choice (1-2): ')

        if choice =='1':

            segmented_organoids_folder = input('Enter folder with segmented organoids: ')
            original_image_folder = input('Enter folder with original images: ')
            cell_surface_channel = input('Enter cell surface marker channel (int;): ')
            json_output_path = input('Enter output path for json file: ')
            
            run_cell_surface(segmented_organoids_folder,
                            original_image_folder,
                            cell_surface_channel,
                            json_output_path)
            
        elif choice =='2':

            print('Returning to main menu')
            break
        
        else:
            print("Invalid Input. Only numerical options allowed.")   

 
if __name__ == "__main__":
    main_menu()


