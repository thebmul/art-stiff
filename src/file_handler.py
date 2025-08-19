"""
This module contains functions to handle DICOM files, specifically for loading IVUS images
and extracting pulse pressure waveform data.
"""

import pydicom
import numpy as np
from PIL import Image
import os


def load_dicom_file(file_path):
    """Loads a DICOM file and returns the dataset."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return
    
    try:
        ds = pydicom.dcmread(file_path)
        return ds
    
    except pydicom.errors.InvalidDicomError:
        print(f"Error: '{file_path}' is not a valid DICOM file.")
        print("It might be a true Unix executable or a corrupted file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def extract_ivus_image(dicom_dataset, frame_index=0):
    """Extracts IVUS image from pixel array from an IVUS DICOM dataset."""
    ## TODO: see if comment necessary \
    try:
        if 'PixelData' in dicom_dataset: # and dicom_dataset.PixelData:
            pixel_array = dicom_dataset.pixel_array
            if hasattr(dicom_dataset, 'NumberOfFrames') and dicom_dataset.NumberOfFrames > 1:
                if not (0 <= frame_index < dicom_dataset.NumberOfFrames):
                    print(f"Warning: frame_index {frame_index} is out of bounds. Displaying first frame (index 0).")
                    frame_index = 0

                if pixel_array.ndim >= 3:
                    ivus_image = pixel_array[frame_index, :, :]
                else:
                    print("Warning: NumberOfFrames > 1 but pixel_array is not 3D. Attempting to display as is.")
                    ivus_image = pixel_array
            else:
                ivus_image = pixel_array
            return ivus_image
        else:
            print("No pixel data found in this DICOM file")
            print("This might be a DICOM file containing only metadata.")
            return None
        
    except IndexError:
        print(f"Error: Could not access frame_index {frame_index}. The DICOM file might not have that many frames.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_dicom_num_frames(dicom_dataset):
    """Returns the number of frames in a DICOM dataset."""
    if hasattr(dicom_dataset, 'NumberOfFrames') and dicom_dataset.NumberOfFrames > 1:
        num_frames = int(dicom_dataset.NumberOfFrames)
        return num_frames
    print("Warning: Assuming single frame.")
    return 1  # Default to 1 frame if not specified

def get_dicom_gain(dicom_dataset, show_success = False):
    """Returns the gain from a DICOM dataset. Returns default value of 50 if gain tag is not found."""
    value = None
    for dataelem in dicom_dataset:  
        tag = str(dataelem.tag)
        if tag == "(0029,1001)":
            value = dataelem.value
            print(f"tag {dataelem.tag} found, value: {dataelem.value}") if show_success else None
    if not value:
        print("Warning: gain tag not found. Assuming default gain setting of 50 dB.")
        value = 50
    return value

def load_binary_mask(filepath, hide_errors=True):
    """Returns a 2D binary NumPy array mask given the file path."""
    try:
        # load the mask file
        img = Image.open(filepath)

        # convert to greyscale, and store as 2D NumPy array
        mask = np.array(img.convert('L'))
    
        # binarize mask 
        mask[mask <= 2] = 1
        mask[mask > 2] = 0
        
        return mask
    
    except FileNotFoundError:
        print(f"Error: '{filepath[(filepath.rfind('/')+1):]}' is not a valid image file.") if not hide_errors else None
        print(f"It might be corrupted, or not exist within the current folder ('{filepath[(filepath.rfind('/', 0, (filepath.rfind('/masks/')))+1):(filepath.rfind('/'))]}/').") if not hide_errors else None
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



## !!!! ##
def extract_waveform_data(dicom_dataset, channel_name="PULSE"):
    """
    Extracts waveform data for a specific channel (e.g., pulse pressure).
    Assumes a DICOM Waveform object.
    """
    if 'WaveformSequence' in dicom_dataset:
        for waveform in dicom_dataset.WaveformSequence:
            # You'll need to inspect actual DICOMs to find the correct ChannelSource
            # or ChannelLabel for pulse pressure. This is a placeholder.
            for channel in waveform.ChannelDefinitionSequence:
                if channel.ChannelLabel == channel_name or channel.ChannelSource.CodeMeaning == channel_name:
                    # Raw waveform data is typically bytes, need to convert to array
                    # The conversion depends on WaveformBitsAllocated and WaveformSampleInterpretation
                    # This is a simplified example.
                    waveform_data = np.frombuffer(waveform.WaveformData, dtype=np.int16) # Assuming int16
                    sampling_frequency = waveform.SamplingFrequency
                    return waveform_data, sampling_frequency
    return None, None
def get_dicom_time(dicom_dataset):
    """Extracts acquisition time from a DICOM dataset."""
    if 'AcquisitionTime' in dicom_dataset:
        return dicom_dataset.AcquisitionTime
    return None
## !!!! ##

"""
if __name__ == '__main__':
    # Simple test for dicom_handler 
    ## import testdata_file from wherever
    file_path = ## file path
    ds = load_dicom_file(file_path)
    if ds:
        print("Loaded DICOM")
        img = extract_ivus_image(ds)

        # Waveform data is less common in general CT, you'd need a specific file
        wave_data, freq = extract_waveform_data(ds, "PULSE")
        if wave_data is not None:
           print(f"Waveform data points: {len(wave_data)}, Freq: {freq}")
           """