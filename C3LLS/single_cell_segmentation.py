import SimpleITK as sitk
import os
import re
from scipy.fftpack import fftn,ifftn,fftshift,ifftshift
from kneed import KneeLocator
from skimage.measure import label,regionprops
from skimage.morphology import remove_small_objects
import numpy as np

def get_2d_power_spectrum(image):

    image  = image.astype(float) 
    fft_image = fftn(image)
    fft_image_shifted = fftshift(fft_image)
    power_spectrum = np.abs(fft_image_shifted)**2
    return power_spectrum

def radial_average_high_pass(power_spectrum):
    h,w = power_spectrum.shape
    center = (h //2 , w //2)

    y,x = np.indices((h,w))
    aspect_ratio = w/h

    # compute radial distances from the center
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2 * aspect_ratio**2)

    radial_indices = np.argsort(radius.flat)
    radial_distances = radius.flat[radial_indices]
    sorted_power_spectrum = power_spectrum.flat[radial_indices]

    nonzero_indices = sorted_power_spectrum > 0 #remove zero power cause it contains nothing meaningful
    radial_distances = radial_distances[nonzero_indices]
    sorted_power_spectrum = sorted_power_spectrum[nonzero_indices]

    radial_bins = np.arange(0, np.max(radial_distances),1)
    radial_power = np.zeros_like(radial_bins,dtype=np.float64)
    counts = np.zeros_like(radial_bins)

    #sum the power values at each radius
    for i, r in enumerate(np.floor(radial_distances).astype(int)):
        if r < len(radial_bins):
            radial_power[r] += sorted_power_spectrum[i]
            counts[r] +=1

    counts[counts ==0] = 1 #make sure no division by zero

    radial_power /= counts 

    nonempty_indices = counts > 0
    radial_bins = radial_bins[nonempty_indices]
    radial_power = radial_power[nonempty_indices]

    return radial_bins,radial_power



def detect_knee_high_pass(radial_frequencies,radial_power):

    knee_locator = KneeLocator(radial_frequencies,radial_power,curve='convex',direction='decreasing')

    return knee_locator.knee
    

def apply_knee_detection_high_pass(power_spectrum):

    radial_frequencies, radial_power = radial_average_high_pass(power_spectrum)
    knee_point = detect_knee_high_pass(radial_frequencies,radial_power)

    return knee_point

def apply_high_pass_filter(image,cutoff):

    f_transform = fftn(image)
    f_transform = fftshift(f_transform)

    # Create high-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = cutoff
    
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    mask = distance >= radius # high pass filter
    f_transform *= mask
    
    # Shift zero frequency component back
    f_transform_filtered = ifftshift(f_transform)
    
    # Perform inverse 2D Fourier transform
    filtered_image = np.abs(ifftn(f_transform_filtered))
    
    return filtered_image    



# assumes that image has been grayscaled already 
# typically use nuclear stain, but you can use any channel you want to segment 
# alternatively you can combine channels to grayscale as you would like

def parse_or_default(value, cast_type, default):
    try:
        return cast_type(value) if value.strip() else default
    except ValueError:
        print(f"Invalid input '{value}'. Using default = {default}")
        return default

def run_single_cell_segmentation(filepath: str,
                                 output_path: str,
                                 minimum_size_input: str,
                                 percentile_input: str ):

    percentile = parse_or_default(percentile_input, float, 95.0)
    minimum_size = parse_or_default(minimum_size_input, int, 0)


    grayscale_image = sitk.ReadImage(filepath)
    grayscale_image = sitk.GetArrayFromImage(grayscale_image)


    #going to populate by z layer
    final_image = np.zeros_like(grayscale_image)

    num_z_layers = grayscale_image.shape[0]

    for z in range(num_z_layers):
        grayscale_z = grayscale_image[z,:,:]
        power_spectrum = get_2d_power_spectrum(grayscale_z)
        knee_frequency_high_pass = apply_knee_detection_high_pass(power_spectrum)
        high_passed_image = apply_high_pass_filter(grayscale_z,cutoff=knee_frequency_high_pass)

        #threshold the layer based on the threshold defined above 
        final_image[z,:,:] = np.where(high_passed_image > np.percentile(high_passed_image,percentile),1,0)

    if minimum_size_input == '':
        labeled_array = label(final_image,connectivity=2)
        regions = regionprops(labeled_array)
        volume_list = []

        for region in regions:
            volume_list.append(region.area)

        volume_array = np.array(volume_list)
        average_volume = np.mean(volume_array)
        standard_dev = np.std(volume_array)

        min_volume = average_volume + standard_dev

    else:
        min_volume = minimum_size

    labeled_array_filtered = remove_small_objects(labeled_array,min_size =min_volume)
    binary_mask = (labeled_array_filtered > 0).astype(np.uint8)

    binary_mask = sitk.GetImageFromArray(binary_mask)
    sitk.WriteImage(binary_mask,output_path)

    return

