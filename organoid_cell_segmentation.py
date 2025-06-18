import os
import SimpleITK as sitk
from scipy.fftpack import fftn, fftshift
from kneed import KneeLocator

image_path = ''
output_path =''
percentile = ''
frequencies = ''
sigma = ''

#You should first segment images using organoid_segmentation.py and running through nnUNetv2, and crop_images.py
#place sample image into a folder and use this script to autosegment it

#function for computing the 3d power spectrum of an image

def get_3d_power_spectrum(image):

  image = image.astype(float) #ensure that we have a float
  fft_image = fftn(image) #get the Fourier transform of the image and shift it 
  fft_image = fftshift(fft_image) 
  power_spectrum = np.abs(fft_image)**2 #square the shifted fft to get the power spectrum

  return power_spectrum

#helper function for identifying the knee point of the power spectrum

def radial_average_3d(power_spectrum):
   """
    Computes the radial average of a 3D power spectrum while maintaining an anisotropic scaling factor for z.
    Do this because most images have a much smaller z dimnesion that x,y
    """

  dz,dy,dx = power_spectrum.shape
  center = (dz // 2, dy //2 , dx //2)
  z_scale = (dz/max(dy,dx)) # adjust for the aspect ratio
  z,y,x = np.indices([dz,dy,dx])
  aspect_ratio = dx/dy 

  # compute radial distances
  radius = np.sqrt((x - center[2]) ** 2 + ((y - center[1])* aspect_ratio) ** 2 + ((z - center[0]) * z_scale) ** 2)

  #flatten and sort power spectrum by radial distances
  radial_indices = np.argsort(radius.flat)
  radial_distances = radius.flat[radial_indices]
  sorted_power_spectrum = power_spectrum.flat[radial_indices]

     # Filter out zero power values
    nonzero_indices = sorted_power_spectrum > 0
    radial_distances = radial_distances[nonzero_indices]
    sorted_power_spectrum = sorted_power_spectrum[nonzero_indices]

    # Compute radial bins and average power per bin
    radial_bins = np.arange(0, np.max(radial_distances), 1)
    radial_power = np.zeros_like(radial_bins, dtype=np.float64)
    counts = np.zeros_like(radial_bins)

    for i, r in enumerate(np.floor(radial_distances).astype(int)):
        radial_power[r] += sorted_power_spectrum[i]
        counts[r] += 1

    counts[counts == 0] = 1  # Avoid division by zero
    radial_power /= counts  # Normalize

    return radial_bins, radial_power

  
#once you have the radial frequencies and power you can find at which frequency does power drop off

def detect_knee_3d(radial_frequencies,radial_power):

  nonzero_power_indices = radial_power > 0 #remove zero power regions b/c they are uninformative and can mess with knee detection
  filtered_frequencies = radial_frequencies[nonzero_power_indices]
  filtered_power = radial_power[nonzero_power_indices]

  knee_locator = KneeLocator(filtered_frequencies,filtered_power,curve='convex',direction='decreasing')
  return knee_locator.knee
  

def apply_knee_detection_3d(power_spectrum):

  radial_frequencies, radial_power = radial_average_3d(power_spectrum)
  knee_point = detect_knee_3d(radial_frequencies,radial_power)
  
  return knee_point
  



grayscale_image = sitk.ReadImage(image_path)
grayscale_image = sitk.GetArrayFromImage(grayscale_image)
grayscale_image = grayscale_image.astype(np.float32) #important for FFT

num_z_layers = grayscale_image.shape[0]

#obtain the power spectrum for the image
#use the power spectrum and a knee point algorithm to identify the cutoff

power_spectrum = get_3d_power_spectrum(grayscale_image)
knee_frequency_high_pass = loadFunctions.apply_knee_detection_3d(power_spectrum)


# # # Apply the filter in the frequency domain
fft_img = fftshift(fftn(grayscale_image))
# filtered_fft = fft_img * log_gabor
# filtered_img = np.abs(ifftn(fftshift(filtered_fft)))

max_freq = loadFunctions.get_max_frequency_cutoff(grayscale_image.shape)
frequencies = np.arange(1,max_freq/7,1)



low_pass_image = loadFunctions.apply_gaussian_low_pass_filter_3d(grayscale_image,cutoff=knee_frequency_high_pass)
background_binary_mask = np.zeros_like(low_pass_image)
foreground_binary_mask = np.zeros_like(low_pass_image)



for z in range(num_z_layers):
    layer_threshold = threshold_otsu(low_pass_image[z,:,:])
    background_binary_mask[z,:,:] = low_pass_image[z,:,:] < layer_threshold
    foreground_binary_mask[z,:,:] = low_pass_image[z,:,:] > layer_threshold


sigma_values = np.arange(0.1, 2, .1)  # Test different sigma_f values
best_snr = 0


# for sigma_f in sigma_values:
#     _, snr_values = loadFunctions.compute_snr(grayscale_image, background_binary_mask, frequencies, sigma_f=sigma_f)
#     avg_snr = np.mean(snr_values)  # Take the mean SNR across frequencies
    
#     if avg_snr > best_snr:
#         best_snr = avg_snr
#         best_sigma = sigma_f

best_sigma = .5

# Compute SNR across frequencies
#freqs, snr_values = loadFunctions.compute_snr(grayscale_image, background_binary_mask, frequencies, sigma_f=best_sigma)

#frequencies = freqs[snr_values > 0]  # Choose frequencies where SNR > 1

phase_response = []



for f0 in frequencies:

    log_gabor = loadFunctions.log_gabor_3d_filter(grayscale_image.shape, f0, sigma_f=best_sigma)
    filtered_fft = fft_img * log_gabor
    filtered_img = ifftn(filtered_fft)
    phase_response.append(np.angle(filtered_img))  # Extract phase

phase_sum = np.sum(np.exp(1j * np.array(phase_response)), axis=0)
phase_congruency_map = np.abs(phase_sum) / len(phase_response)  # Normalize
threshold = np.percentile(phase_congruency_map,percentile)
phase_congruency_map_binary = phase_congruency_map > threshold
