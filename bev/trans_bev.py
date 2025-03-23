from BirdsEyeView import BirdsEyeView
from glob import glob
import os, sys
import cv2
import numpy as np


def main(dataFiles, pathToCalib, outputPath, calib_end='.txt'):
    '''
    Main method of transform2BEV
    :param dataFiles: the files you want to transform to BirdsEyeView, e.g., /home/elvis/kitti_road/data/*.png
    :param pathToCalib: containing calib data as txt-files, e.g., /home/elvis/kitti_road/calib/
    :param outputPath: where the BirdsEyeView data will be saved, e.g., /home/elvis/kitti_road/data_bev
    :param calib_end: file extension of calib-files (OPTIONAL)
    '''

    # Extract path of data
    pathToData = os.path.split(dataFiles)[0]
    assert os.path.isdir(pathToData), "The directory containig the input data seems to not exist!"
    assert os.path.isdir(pathToCalib), "Error <PathToCalib> does not exist"

    # BEV class
    bev = BirdsEyeView()

    # check
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    # get filelist
    fileList_data = glob(dataFiles)
    assert len(fileList_data), 'Could not find files in: %s' % pathToData

    # Loop over all files
    for aFile in fileList_data:
        assert os.path.isfile(aFile), '%s is not a file' % aFile

        file_key = aFile.split('/')[-1].split('.')[0]
        print("Transforming file %s to Birds Eye View " % file_key)
        tags = file_key.split('_')
        data_end = aFile.split(file_key)[-1]

        # calibration filename
        calib_file = os.path.join(pathToCalib, file_key + calib_end)

        if not os.path.isfile(calib_file) and len(tags) == 3:
            # exclude lane or road from filename!
            calib_file = os.path.join(pathToCalib, tags[0] + '_' + tags[2] + calib_end)

        # Check if calb file exist!
        if not os.path.isfile(calib_file):
            print("Cannot find calib file: %s" % calib_file)
            print("Attention: It is assumed that input data and calib files have the same name (only different extension)!")
            sys.exit(1)

        # Update calibration for Birds Eye View
        bev.setup(calib_file)

        # Read image
        # data = cv2.imread(aFile, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # data = cv2.imread(aFile, 0)
        data = cv2.imread(aFile)

        # label_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # data = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
        # data[label_image[:, :, 2] > 0] = 255

        # Compute Birds Eye View
        data_bev = bev.compute(data)

        # Write output (BEV)
        fn_out = os.path.join(outputPath, file_key + data_end)
        if (cv2.imwrite(fn_out, data_bev)):
            print("done ...")
        else:
            print("saving to %s failed ... (permissions?)" % outputPath)
            return

    print("BirdsEyeView was stored in: %s" % outputPath)


if __name__ == "__main__":

    dataFiles = '/home/changyicong/Data/KITTI/testing/image_2/*.png'
    pathToCalib = '/home/changyicong/Data/KITTI/testing/calib/'
    outputPath = '/home/changyicong/Data/KITTI_BEV/testing/image_2/'

    # Excecute main fun
    main(dataFiles, pathToCalib, outputPath)
