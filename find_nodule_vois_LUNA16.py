# This code goes thru each dataset in LUNA16 and save individual image slices  as numpy arrays, each a with known
# annotated nodule object in the center of the img. We will hopefully use this to find a reasonable threshold
# (in Hounsfield Units) when binaryzing the kaggle datasets when trying to find the best X number of nodule candidates
#  to use in training a CNN.

# This code is a modified version of LUNA_mask_extraction.py from https://github.com/booz-allen-hamilton/DSB3Tutorial

from __future__ import print_function, division

import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
from skimage.measure import regionprops
from skimage import measure, morphology

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    # x_data = [x*spacing[0]+origin[0] for x in range(width)]
    # y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    # return(mask)

    # label the distinct connected sphere object in created mask img
    labels = measure.label(mask)
    # if labels == 0:
    #     print("No labeled object, something went amiss (see coord")
    # Measure properties of labeled region
    label_measures = regionprops(labels)
    print(len(label_measures))
    if len(label_measures) == 0:
        return (0,0,0,0)
    print(label_measures[0].bbox)
    (min_row, min_col, max_row, max_col) = label_measures[0].bbox

    return (min_col, max_col, min_row, max_row)

def matrix2int16(matrix):
    '''
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


if __name__ == '__main__':

    # Getting list of datasets (local)
    # luna_path = "/home/shonket/python_code/kaggle/kaggle_databowl_17/LUNA16"
    # subset_list = ["/subset{}".format(str(x)) for x in range(2)]
    # output_path = "/home/shonket/python_code/kaggle/kaggle_databowl_17/LUNA16/nodule_vois"

    # Getting list of datasets (local)
    luna_path = "/home/ec2-user/LUNA16"
    subset_list = ["/subset{}".format(str(x)) for x in range(10)]
    output_path = "/home/ec2-user/LUNA16/nodule_slices"

    # loop thru each subset (10 total) in LUNA16
    for subset in subset_list:
        # each img set in one mhd file (kaggle img set has one file PER individual img)
        full_path = luna_path + subset
        # print(full_path)
        file_list = glob(full_path + "/*.mhd")
        # print(file_list)

        # The locations of the nodes
        df_node = pd.read_csv(luna_path + "/CSVFILES/annotations.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        df_node = df_node.dropna()
        print("for {}, # of nodules is {}".format(subset, df_node.shape[0]))

        slice_fails_list = []

        # loop thru each img set
        for fcount, img_file in enumerate(file_list):
            mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
            if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
                # load the data once
                itk_img = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
                num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
                origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
                spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
                # go through all nodes
                for node_idx, cur_row in mini_df.iterrows():
                    print('Node index is {}'.format(node_idx))
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]
                    print('mm coord {}, {}, {}, {}'.format(node_x,node_y,node_z,diam))
                    # just keep 3 slices for each node
                    # imgs = np.ndarray([3, height, width], dtype=np.float32)
                    # masks = np.ndarray([3, height, width], dtype=np.uint8)
                    center = np.array([node_x, node_y, node_z])  # nodule center
                    v_center = np.rint(
                        (center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                    print('voxel center is {}'.format(v_center))
                    # take slice on v_center with slice above and below
                    for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                      int(v_center[2]) + 2).clip(0,
                                                                                 num_z - 1)):  # clip prevents going out of bounds in Z
                        # mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                        #                  width, height, spacing, origin)
                        # masks[i] = mask
                        (v_xmin, v_xmax, v_ymin, v_ymax) = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                        if (v_xmin, v_xmax, v_ymin, v_ymax) == (0,0,0,0):
                            print('failed, skip')
                            slice_fails_list.append(os.path.join(output_path, "images_%04d_%04d_%04d.npy" % (fcount, node_idx, i)))
                            continue

                        # make 2d slice contained node and save np array file per slice
                        temp_img = img_array[i_z, v_ymin:v_ymax, v_xmin:v_xmax]
                        np.save(os.path.join(output_path, "images_%04d_%04d_%04d.npy" % (fcount, node_idx, i)), temp_img)

                    # np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, node_idx)), masks)

    print('These ones failed for some reason:\n')
    for item in slice_fails_list:
        print(item)
