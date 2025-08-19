import os
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import train_test_split
from file_handler import load_dicom_file, get_dicom_num_frames, extract_ivus_image, get_dicom_gain, load_binary_mask
from image_processor import normalize_image, resize_matrix, expand_along_last_axis, random_alteration
from u_net_model import initialize_model, train_and_save_model, load_model
# Import other modules as you create them:
# from image_processor import normalize_image, resize_matrix, expand_along_last_axis, random_alteration
#from u_net_model import *
# from src.waveform_analyzer import detect_pressure_peaks
# from src.synchronizer import synchronize_data
# from src.stiffness_calculator import calculate_pressure_gradient, calculate_distensibility, calculate_compliance, calculate_stiffness_index

def load_masks_and_images(studies_path, show_print_debug_statements=False):
    """
    Function to load masks and images.
    """
    # initialize lists of:
    #     images (X)  
    all_usable_images = []
    # and
    #     masks (y)
    all_usable_masks = []

    # initialize list of images to use for visual inspection of model performance
    all_other_images = []

    # list of all study folders in studies-usable
    studies_list = [f for f in os.listdir(studies_path) if not f.endswith('.DS_Store')]
    # iterate through the list of study folders
    for study_name in studies_list:
        # path for the study folder
        study_path = os.path.join(studies_path, study_name)
        
        # path for the DICOM image
        dicom_file_name = study_name[(study_name.rfind('-')+1):]
        dicom_file_path = os.path.join(study_path, dicom_file_name)

        # path for mask folder
        mask_folder_path = os.path.join(study_path, "masks")

        # load DICOM file
        ds = load_dicom_file(dicom_file_path)

        if ds:        
            matrices_dict = {}
            # extract DICOM image metadata
            num_frames = get_dicom_num_frames(ds)
            gain = get_dicom_gain(ds)

            # iterate through each frame in the DICOM image
            for frame_idx in range(num_frames):
                frame_num = frame_idx + 1

                # load and normalize image
                ivus_image = extract_ivus_image(ds, frame_index = frame_idx)
                norm_image = normalize_image(ivus_image)

                # path for mask file corresponding to specific DICOM frame
                mask_file_name = 'Mask' + str(frame_num) + '.png'
                mask_file_path = os.path.join(mask_folder_path, mask_file_name)

                # load mask file as a binary NumPy array
                binary_mask = load_binary_mask(mask_file_path)

                if binary_mask is not None:
                    altered_image, altered_mask, move, new_gain = random_alteration(image=norm_image, mask=binary_mask, gain=gain, show_internals = False)
                    all_usable_images.append(expand_along_last_axis(resize_matrix(altered_image)).astype(np.float32))
                    all_usable_masks.append(resize_matrix(altered_mask))
                else:
                    print(f"No mask associated with {dicom_file_name} frame {frame_num}/{num_frames}.") if show_print_debug_statements else None
                    print("Storing unaltered image in 'all_other_images' list.") if show_print_debug_statements else None
                    all_other_images.append(norm_image)

    return all_usable_images, all_usable_masks, all_other_images

def process_and_save_data(all_usable_images, all_usable_masks, processed_data_folder_path, attempt_folder_str, show_print_debug_statements=False):
    """
    
    """
    allimages = np.array(all_usable_images)
    allmasks = np.array(all_usable_masks)

    if show_print_debug_statements:
        print("\nIVUS Images")
        print(f"Shape: {allimages.shape}")
        print(f"Data Type: {allimages.dtype}")
        print("\nMasks")
        print(f"Shape: {allmasks.shape}")
        print(f"Data Type: {allmasks.dtype}")

    #90% train, 10% test
    X_train, X_test, y_train, y_test = train_test_split(allimages, allmasks, test_size=0.1, random_state=42)

    #80% train, 10% test, 10% validation
    one_ninth = 1/9
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=one_ninth, random_state=42)
    
    if show_print_debug_statements:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        # print()


    # save the data

    # DENOTE ATTEMPT FOLDER! DO NOT OVERWRITE!
    # = 'attempt-one/'

    print("Initiating process of saving Test-Train-Split data... ") if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_train.npy', X_train)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'Unet-IVUS-images-X_train.npy\'') if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_train.npy', y_train)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'Unet-IVUS-images-y_train.npy\'') if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_val.npy', X_val)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'Unet-IVUS-images-X_val.npy\'') if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_val.npy',y_val)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'/net-IVUS-images-y_val.npy\'') if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_test.npy', X_test)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'Unet-IVUS-images-X_test.npy\'') if show_print_debug_statements else None
    np.save(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_test.npy', y_test)
    print('saved \'/Users/brianmulhern/Documents/dev/arterial-stiffness/art-stiff/data/processed/'+attempt_folder_str+'Unet-IVUS-images-y_test.npy\'') if show_print_debug_statements else None
    print("Finished saving the data ...") if show_print_debug_statements else None
    
def load_data(processed_data_folder_path, attempt_folder_str, show_print_debug_statements=False):
    """
    Loads the processed data from the specified folder path and attempt folder string.
    Returns the training, validation, and test datasets.
    """
    X_train = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_train.npy')
    y_train = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_train.npy')
    X_val = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_val.npy')
    y_val = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_val.npy')
    X_test = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-X_test.npy')
    y_test = np.load(processed_data_folder_path+attempt_folder_str+'Unet-IVUS-images-y_test.npy')
    
    if show_print_debug_statements:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        # print()

    return X_train, y_train, X_val, y_val, X_test, y_test
 
def set_model_params():
    """
    Placeholder function to set model parameters.
    This will be replaced with actual model configuration logic.
    """
    # Example parameters

"""
def run_analysis(dicom_folder_path):
    
    Main function to orchestrate the IVUS and pulse pressure analysis.
    
    ivus_files = [f for f in os.listdir(dicom_folder_path)] # if f.endswith('.dcm')]
    ivus_data = {} # Store image data, metadata, and placeholder for processed results


    for filename in ivus_files:
        filepath = os.path.join(dicom_folder_path, filename)
        ds = load_dicom_file(filepath)
        if ds:
            # Extract IVUS image
            num_frames = get_dicom_num_frames(ds)
            
            ivus_image = extract_ivus_image(ds, frame_index = num_frames)

            # Check and record the number of frames
            #num_frames = get_dicom_num_frames(ds)
            
            ## TODO: Handle timing / synchronization issues
            ###      i.e. there's no waveform data to sync up...
            # Extract time for synchronization
            ## acquisition_time = get_dicom_time(ds)

            # Placeholder for processing steps
            p_sys, p_dia = #extract_systolic_diastolic_pressures(pulse_waveform, sampling_freq)
            delta_p = calculate_pressure_gradient(p_sys, p_dia)

            # end_systole_img, end_diastole_img = identify_phases(ivus_image_sequence)
            area_sys = measure_lumen_area(end_systole_img)
            area_dia = measure_lumen_area(end_diastole_img)
            distensibility = calculate_distensibility(area_sys, area_dia, delta_p)
            compliance = calculate_compliance(area_sys, area_dia, delta_p)
            stiffness_index = calculate_stiffness_index(area_sys, area_dia, delta_p)

            ### ISSUE: assumes all files have unique names
            ivus_data[filename] = {
                'dataset': ds,
                'image': ivus_image,
                ## 'acquisition_time': acquisition_time,
                # 'es_area': es_area,
                # 'ed_area': ed_area,
            }

    

    # Assuming you'll have one or more DICOM files with waveform data
    # This part needs to be more robust for finding the correct waveform file
    waveform_file = None # You'll need logic to identify the correct waveform DICOM
    # For now, let's assume it's also in the dicom_folder_path and you find it.
    for filename in ivus_files: # Or look for a dedicated waveform file
        if "WAVEFORM" in filename.upper(): # A simple heuristic
             waveform_file = os.path.join(dicom_folder_path, filename)
             break

    pulse_pressure_data = None
    if waveform_file:
        ds_waveform = load_dicom_file(waveform_file)
        if ds_waveform:
            pulse_waveform, sampling_freq = extract_waveform_data(ds_waveform, "PULSE")
            # pulse_systolic_peaks, pulse_diastolic_peaks = detect_pressure_peaks(pulse_waveform, sampling_freq)
            pulse_pressure_data = {
                'waveform': pulse_waveform,
                'sampling_frequency': sampling_freq,
                'acquisition_time': get_dicom_time(ds_waveform),
                # 'systolic_peaks': pulse_systolic_peaks,
                # 'diastolic_peaks': pulse_diastolic_peaks
            }


    # Synchronization and Stiffness Calculation
    if ivus_data and pulse_pressure_data:
        print("Data loaded for IVUS images and pulse pressure waveform.")
        # synchronized_measurements = synchronize_data(ivus_data, pulse_pressure_data)
        # stiffness_results = calculate_stiffness(synchronized_measurements)
        # print("Arterial stiffness calculated:", stiffness_results)
    else:
        print("Could not load sufficient data for analysis.")
"""
# bullshit above, ignore

if __name__ == '__main__':
    load_dotenv()
    filepath = os.getenv("FILEPATH")

    # load masks and images
    studies_data_path = filepath+'data/studies-usable/'
    building_images, building_masks, visual_test_images = load_masks_and_images(studies_data_path)

    # process and save data
    processed_data_path = filepath+'data/processed/'
    attempt_str = 'attempt-one/'  # or 'attempt-two/' etc.
    os.mkdir(processed_data_path+attempt_str) if not os.path.exists(processed_data_path+attempt_str) else None
    process_and_save_data(building_images, building_masks, processed_data_path, attempt_str)
  
    # load data for training
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(processed_data_path, attempt_str)

    parameters = {
        'model name': 'unet_ivus_model_one',
        # dropout parameters
        'do dropout': True,
        'dropout rate': 0.04,

        # model architecture parameters
        'filters': [16,32,48,64],
        # model compilation parameters
        'block activation': "relu",
        'output activation': "sigmoid",
        'optimizer title': "Adam",
        'learning rate': 5e-5,
        # loss function parameters
        'lambda binary crossentropy loss': 1.0,
        'lambda dice loss': 0.0,
        # callbacks parameters
        'learn rate intervention monitor': "val_loss", # "val_loss" or "val_accuracy"
        'plateau patience': 3, 
        'learn rate reduction factor': 0.1,
        'minimum learn rate': 1e-6,
        'early stopping intervention monitor': "val_loss", # "val_loss" or "val_accuracy"
        'early stopping patience': 15,
        # training parameters
        'batch size': 8,
        'epochs': 50
    }

    # initialize, train, and save model
    model = initialize_model(parameters, show_summary=True)
    trained_models_path = filepath+'models/'
    history = train_and_save_model(parameters, X_train, y_train, X_val, y_val, model, trained_models_path, show_training_process=True)
    
    # load model
    # model = load_model(trained_models_path + parameters['model name'] + '.keras')