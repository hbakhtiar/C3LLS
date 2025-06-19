import SimpleITK as sitk
import os
import re
from scipy.fftpack import fftn,ifftn,fftshift,ifftshift
from kneed import KneeLocator

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

    counts[count ==0] = 1 #make sure no division by zero

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
    



# assumes that image has been grayscaled already 
# typically use nuclear stain, but you can use any channel you want to segment 
# alternatively you can combine channels to grayscale as you would like

grayscale_path = ''



grayscale_image = sitk.ReadImage(grayscale_path)
grayscale_image = sitk.GetArrayFromImage(grayscale_image)


#going to populate by z layer
final_image = np.zeros_like(grayscale_image)

num_z_layers = grayscale_image.shape[0]

for z in range(num_z_layers):
    grayscale_z = grayscale_image[z,:,:]
    power_spectrum = get_2d_power_spectrum(grayscale_z)
    knee_frequency_high_pass = apply_knee_detection_high_pass(power_spectrum)
