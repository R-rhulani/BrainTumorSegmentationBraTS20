import numpy as np
import nibabel as nib

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
import logging
logging.basicConfig(filename='corrupted_files.log',level=logging.DEBUG)

TRAIN_DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

myArray = []
fo = open("corruptFilesList.txt", "r")
for line in fo:
    for nbr in line.split():
        myArray.append(int(nbr))
rangeStart = 10

for file in range(rangeStart, 70):
    fileNumber = '3' + str(file)
    if int(fileNumber) in myArray:
        print('files in folder '+ fileNumber + ' are corrupted')
    else:
        try:
            test_image_flair = nib.load(
                TRAIN_DATASET_PATH + 'BraTS20_Training_' + fileNumber + '/BraTS20_Training_' + fileNumber + '_flair.nii').get_fdata()
            print('test_image_flair.max() for file number ' + fileNumber + ' is :')
            test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(
                test_image_flair.shape)

            test_image_t1 = nib.load(
                TRAIN_DATASET_PATH + 'BraTS20_Training_' + fileNumber + '/BraTS20_Training_' + fileNumber + '_t1.nii').get_fdata()
            test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(
                test_image_t1.shape)

            test_image_t1ce = nib.load(
                TRAIN_DATASET_PATH + 'BraTS20_Training_' + fileNumber + '/BraTS20_Training_' + fileNumber + '_t1ce.nii').get_fdata()
            test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(
                test_image_t1ce.shape)

            test_image_t2 = nib.load(
                TRAIN_DATASET_PATH + 'BraTS20_Training_' + fileNumber + '/BraTS20_Training_' + fileNumber + '_t2.nii').get_fdata()
            test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(
                test_image_t2.shape)

            test_mask = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_' + fileNumber + '/BraTS20_Training_' + fileNumber + '_seg.nii').get_fdata()
            test_mask = test_mask.astype(np.uint8)
            print(test_image_flair.max())

        except Exception as e:
            print("test_image_flair.max() for file number "+fileNumber+" has error:" + str (e))
            rangeStart += 1
            logging.debug(int(fileNumber))
            continue

