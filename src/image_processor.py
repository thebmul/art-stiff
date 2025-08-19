"""
This module contains the logic for processing IVUS image and pulse pressure waveform data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, draw 
# from scipy.signal import find_peaks
import random
# from dicom_handler import get_dicom_num_frames

def normalize_image(ivus_image):
    if ivus_image.ndim != 2:
        print("Error: normalize_image expects a 2D image array.")
        return None
    normalized_image = (ivus_image - np.min(ivus_image)) / (np.max(ivus_image) - np.min(ivus_image) + 1e-9) # Add epsilon to avoid division by zero
    return normalized_image
    
def binarize_image(normalized_image, pixel_intensity_thresh_const = 0.15):
    threshold_value = np.min(normalized_image) + (np.max(normalized_image) - np.min(normalized_image)) * pixel_intensity_thresh_const # Pixels in the lowest 15% intensity
    binary_image = normalized_image < threshold_value
    return binary_image
    
def resize_matrix(matrix, border_size=6):
    """
    Takes og_matrix, a 500x500 2-dimensional numpy.ndarray.
    Returns resized_matrix, a 512x512 2-dimensional numpy.ndarray 
    resized_matrix is essentially the original matrix with a 6-row / 6-column border of 0s.
    """
    new_height = matrix.shape[0] + (2 * border_size)
    new_width = matrix.shape[1] + (2 * border_size)
    resized_matrix = np.zeros((new_height, new_width), dtype=matrix.dtype)

    resized_matrix[border_size:-border_size, border_size:-border_size] = matrix
    return resized_matrix
    
def expand_along_last_axis(matrix):
    return np.expand_dims(matrix, axis = -1)
    

def adjust_gain(matrix, old_gain, adj_lim=8):
    if adj_lim > 20:
        print(f"Warning: adj_lim value ({adj_lim}) is greater than the threshold value (20).")
        print("\tadj_lim defaulting to nearest acceptable value (20).")
        adj_lim = 20
    elif adj_lim <= 0:
        print(f"Warning: adj_lim value ({adj_lim}) is less than the threshold value (1).")
        print("\tadj_lim defaulting to nearest acceptable value (1).")
        adj_lim = 1
            
    if old_gain < 40:
        print(f"Warning: old_gain value of {old_gain} is below threshold value of 40.")            
        print(f"\tdefaulting to gain increase of 1 to {adj_lim if adj_lim<10 else 10}.")
        new_gain = np.random.randint(old_gain+1, old_gain+1+adj_lim)
            
    else:
        new_gain = old_gain
        lo = old_gain-adj_lim
        hi = old_gain+adj_lim
        if hi > 68:
            hi = 68
        while new_gain == old_gain:
            new_gain = np.random.randint(lo, hi+1)
            if new_gain == 68:
                break
        
    scalar = pow(10, ((new_gain - old_gain)/20))
    scaled_matrix = matrix * scalar
        
    clipped_matrix = np.clip(scaled_matrix, a_min=None, a_max=1)
        
    return new_gain, clipped_matrix

def no_change(m):
    return m, "orientation unchanged."

def ccw_turn(m):
    return np.rot90(m), "rotated counter-clockwise."

def turn_180(m):
    return np.rot90(m, k=2), "rotated 180 degrees."
    
def cw_turn(m):
    return np.rot90(m, k=3), "rotated clockwise."

def flip_along_y(m):
    return np.flipud(m), "flipped about y=0." 

def flip_along_x(m):
    return np.fliplr(m), "flipped about x=0." 

def transposition(m):
    return np.transpose(m), "flipped about y=-x."

def diagonal_flip(m):
    intermediate_m = np.transpose(m)
    return np.rot90(intermediate_m, k=2), "flipped about y=x."


def random_alteration(image, mask, gain, show_internals = False):
    """randomly fucks with all the arrays"""

    pre_processed_mask = mask.copy()

    gain_decision = np.random.randint(0, 2)
            
    if gain_decision == 1:
        print(f"gain_decision = {gain_decision}, so we're fuckin w/ the gain") if show_internals else None
        pre_processed_image = image.copy() 
        gain_adjustment, gain_adjusted_image = adjust_gain(pre_processed_image, gain)

        gain_was_adjusted = f"Gain adjusted from {gain} to {gain_adjustment}."
    else:
        gain_adjusted_image = image.copy()
        gain_adjustment = gain
        print(f"gain_decision = {gain_decision}, so we're leaving the gain alone") if show_internals else None

    random_ori = random.randint(0, 7)
            
    orientation_moves = [
        lambda m: no_change(m),      # No Change 
        lambda m: ccw_turn(m),       # Rotate 90 degrees counter-clockwise
        lambda m: turn_180(m),       # Rotate 180 degrees
        lambda m: cw_turn(m),        # Rotate 90 degrees clockwise
        lambda m: flip_along_y(m),   # Flip left to right
        lambda m: flip_along_x(m),   # Flip up to down
        lambda m: transposition(m),  # Transpose
        lambda m: diagonal_flip(m)   # Transpose and Rotate 180 degrees
    ]
            
    orientation_function = orientation_moves[random_ori]
    processed_image, move = orientation_function(gain_adjusted_image)
    processed_mask, move = orientation_function(pre_processed_mask)

        #changes = move + " " + gain_was_adjusted if gain_was_adjusted else move
                
        #processed_matrices[matrix_counter] = [changes, processed_matrix]
            
    return processed_image, processed_mask, move, gain_adjustment



class ImageProcessor:
    """
    A class to encapsulate methods for processing IVUS images and masks. 
    The methods are designed to be used with synchronized data from 
    waveform (pulse pressure) analysis.
    """

# TODO: ???   IDK yet
"""
    @staticmethod
    def identify_lumen(
        ivus_image: np.ndarray,
        frame_index: int = 0,
        eccentricity_threshold: float = 0.9999,
        pixel_intensity_thresh_const: float = 0.15,
        perform_small_obj_removal: bool = False,
        small_obj_removal_min_size: int = 500,
        perform_binary_closing: bool = False,
        binary_closing_disk_size: int = 5,
        min_rel_area: float = 0.005,
        max_rel_area: float = 0.25
    ) -> tuple:
        
    ###
        Identifies the lumen in an IVUS image using image processing techniques.

        Args:
            ivus_image (np.ndarray): The IVUS image array.
            frame_index (int): The frame index of the IVUS image being analyzed. 
                    Default value of 0.
            eccentricity_threshold (float): Adjusts the threshold of the 
                    acceptable eccentricity of the region identified as the lumen. 
                    0.0 is a perfect circle. Default value of 0.875.
            pixel_intensity_thresh_const (float): Adjusts the constant for pixel 
                    intensity threshold calculation used to identify the vascular 
                    media (the outer boundary of the vascular lumen) during the 
                    normalization operation. A higher value will result in a larger 
                    area being classified as the media. Default value of 0.15.
            perform_small_obj_removal (bool): Whether or not to remove small 
                    objects from the binary image, which is pretty self explanatory. 
                    Default value of False.
            small_obj_removal_min_size (int): Adjusts the threshold of the minimum
                    size of the small objects to be removed from the binary image. 
                    structuring element in the binary closing operation. A larger
                    value will produce a more aggressive removal of small objects.
                    Default value of 500.
            perform_binary_closing (bool): Whether or not to perform binary 
                    closing, which fills small gaps and smooths the circle's 
                    edges. Default value of False.
            binary_closing_disk_size (int): Adjusts the size of the circular 
                    structuring element in the binary closing operation. A smaller
                    value will produce a more granulae closing effect, resulting 
                    in a more precise lumen boundary. Default value of 5.
            min_rel_area (float): Minimum area of the identified lumen as a
                    percentage of the total image area. Default value of 0.5%.
            max_rel_area (float): Maximum area of the identified lumen as a 
                    percentage of the total image area. Default value of 25%.

        Returns:
            Tuple: A tuple containing:
                - frame (int): the frame / slice / instance of the IVUS image 
                        being analyzed.
                - area (float): The area of the identified lumen in cm^2.
        ###
            # ^^^ change frame to time for sake of synchronization with waveform data   

        if ivus_image.ndim != 2:
            print("Error: identify_lumen expects a 2D image array.")
            return None, None
        
        normalized_image = (ivus_image - np.min(ivus_image)) / (np.max(ivus_image) - np.min(ivus_image) + 1e-9) # Add epsilon to avoid division by zero
        
        threshold_value = np.min(normalized_image) + (np.max(normalized_image) - np.min(normalized_image)) * pixel_intensity_thresh_const # Pixels in the lowest 15% intensity
        binary_image = normalized_image < threshold_value

        if perform_small_obj_removal:
            cleaned_image = morphology.remove_small_objects(binary_image, min_size=small_obj_removal_min_size)
        else:
            cleaned_image = binary_image

        if perform_binary_closing:
            closed_image = morphology.binary_closing(cleaned_image, morphology.disk(binary_closing_disk_size))
        else:
            closed_image = cleaned_image

    
        ## TODO: add more image processing parameters as needed
        #       for example, morphological operations, edge detection, Gaussian Filter???
        # Apply a Gaussian filter to smooth the image
        # smoothed_image = filters.gaussian(ivus_image, sigma=1)


        labels = measure.label(closed_image)
        regions = measure.regionprops(labels)

        best_shape = None

        min_distance_to_center = float('inf')
        image_center_y, image_center_x = np.array(ivus_image.shape) / 2

        for region in regions:
            # Filter out very small or very large regions that are unlikely to be the circle
            # Adjust min_area and max_area based on expected circle size in your images
            min_area = (ivus_image.shape[0] * ivus_image.shape[1]) * min_rel_area 
            max_area = (ivus_image.shape[0] * ivus_image.shape[1]) * max_rel_area
            
            if region.area < min_area or region.area > max_area:
                continue

            # Filter for regions that are reasonably circular (e.g., circularity > 0.7 or low eccentricity)

            # Calculate circularity: 1 for a perfect circle, lower for irregular shapes
            # Avoid division by zero if perimeter is 0
            # circularity = 4 * np.pi * region.area / (region.perimeter**2) if region.perimeter > 0 else 0
            
            # Eccentricity is 0 for a circle, 1 for a line segment
            if region.eccentricity > eccentricity_threshold: # Adjust this value as needed, lower is more circular
                continue

            # Calculate distance of region centroid to image center
            distance_to_center = np.linalg.norm(np.array(region.centroid) - np.array([image_center_y, image_center_x]))

            # We're looking for the most centrally located region that fits the criteria
            if best_shape is None or distance_to_center < min_distance_to_center:
                best_shape = region
                min_distance_to_center = distance_to_center

        if best_shape:
            # For ovals, it's more appropriate to return major_axis_length or equivalent diameter
            # If you need precise oval dimensions, you'd use region.major_axis_length and region.minor_axis_length
            # For drawing a circle that approximates the oval, we can use the equivalent diameter
            estimated_radius = np.sqrt(best_shape.area / np.pi) # Equivalent radius of a circle with the same area
            return best_shape.centroid, estimated_radius
        else:
            return None, None
        
    @staticmethod
    def record_lumen_radii(
        num_frames: int,
        ivus_image: np.ndarray
    ) -> pd.core.frame.DataFrame:
        ###
        Records the areas of the identified lumen in an IVUS image.

        Args:
            num_frames (int): The number of frames in the IVUS image.
            ivus_image (np.ndarray): The IVUS image array.

        Returns:
            pd.DataFrame: A DataFrame containing the frame index and area of the 
                    relative identified lumen.
        ###

        measurements = []

        for i in range(num_frames):
            frmidx_iter = i
            measurements.append(ImageProcessor.identify_lumen(ivus_image, frame_index=frmidx_iter))

        df = pd.DataFrame(measurements, columns=['frame', 'radius'])
        nan_percentage = (df['radius'].isna() / num_frames) * 100

        # Calculate area in cm^2
        #area = np.pi * (radius ** 2)
        
    @staticmethod
    def identify_peaks(
        radii_df: pd.core.frame.DataFrame,
        sampling_frequency: float = 5.0,
        maxima_height_period_shift: int = 2,
        minima_height_period_shift: int = -5,

    ) -> tuple:
        ###
        identifies the Measures the luminal area at peak systole and end diastole from the IVUS image.

        Args:
            ivus_image (np.ndarray): The IVUS image array.

        Returns:
            tuple: A tuple containing:
                - mean_area_sys (float): The mean luminal area at peak systole (cm^2).
                - mean_area_dia (float): The mean luminal area at end diastole (cm^2).
        ###

        
        area_sys = 5.0
        """