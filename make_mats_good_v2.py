import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
#import matplotlib.pyplot as plt
import json
import math
from scipy import sparse

from skimage import measure, morphology
from datetime import datetime
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


max_x = 500
max_y = 500
max_z = 500

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def pad_zeros(mat):
    if mat.shape[0] < max_x:
        gap = (max_x - mat.shape[0])/2.0

        above_zero_vol = np.zeros((int(math.floor(gap)), mat.shape[1], mat.shape[2]))
        below_zero_vol = np.zeros((int(math.ceil(gap)), mat.shape[1], mat.shape[2]))
        
        mat = np.concatenate([below_zero_vol, np.concatenate([mat, above_zero_vol], axis=0)], axis=0)
                
    if mat.shape[1] < max_y:
        gap = (max_y - mat.shape[1])/2.0
        
        above_zero_vol = np.zeros((mat.shape[0], int(math.floor(gap)), mat.shape[2]))
        below_zero_vol = np.zeros((mat.shape[0], int(math.ceil(gap)), mat.shape[2]))
        
        mat = np.concatenate([below_zero_vol, np.concatenate([mat, above_zero_vol], axis=1)], axis=1)
        
        
    if mat.shape[2] < max_z:
        gap = (max_z - mat.shape[2])/2.0
        
        above_zero_vol = np.zeros((mat.shape[0], mat.shape[1], int(math.floor(gap))))
        below_zero_vol = np.zeros((mat.shape[0], mat.shape[1], int(math.ceil(gap))))
        
        mat = np.concatenate([below_zero_vol, np.concatenate([mat, above_zero_vol], axis=2)], axis=2)

    return mat

def filter_mat(mat):
    CUTOFF=0.4
    b = mat.astype('bool')
    sums = b.sum()
    good_inds = [s for s in sums if s > CUTOFF*mat.shape[0]]
    return mat[:, good_inds]

# Normalization
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# Some constants
INPUT_FOLDER = '/home/ec2-user/data/stage1/'#'../input/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# Open cancer status map
with open('/home/ec2-user/maps/pid_to_cancer_status_map.json', 'r') as f:
    pid_to_cancer_map = json.load(f)

vec_list = []
outcome_vec = []


for patient in patients:

    try:
        outcome_vec.extend([pid_to_cancer_map[patient]])
    except KeyError:
        continue
        
    first_patient = load_scan(INPUT_FOLDER + patient)#patients[0])
    first_patient_pixels = get_pixels_hu(first_patient)

    # Resample pixels
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    print("Shape before resampling\t", first_patient_pixels.shape)
    print("Shape after resampling\t", pix_resampled.shape)

    # Segment lungs
   # segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    

    # Normalize/zero-center
    normed = normalize(segmented_lungs_fill)
    segmented_lungs_fill = None
    z_normed = zero_center(normed)
    normed = None

    padded = pad_zeros(z_normed)
    z_normed = None
    vec_list.extend([sparse.csr_matrix(padded.flatten())])
    
# max_x = max([pat.shape[0] for pat in vec_list])
# max_y = max([pat.shape[1] for pat in vec_list])
# max_z = max([pat.shape[2] for pat in vec_list])

# mat_list = []
# for mat in vec_list:
        
    #mat_list.extend([mat])
    
#flat_mats = [mat.flatten() for mat in mat_list]
out_mat = sparse.vstack(vec_list)#np.stack(flat_mats, axis=0)

out_mat = filter_mat(out_mat)

vers = datetime.now().date().isoformat()

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

save_sparse_csr('/home/ec2-user/matrices/matrix_{}.npz'.format(vers), out_mat)   

#np.save('/home/ec2-user/matrices/matrix_{}.npy'.format(vers), out_mat)

with open('/home/ec2-user/matrices/' + 'outcome_vec_{}.json'.format(vers), 'w+') as f:
    json.dump(outcome_vec, f)