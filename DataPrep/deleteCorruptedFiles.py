import shutil
import os

location = "/Users/rhulanihlungwani/School/Year Project/2023/semanticSegmentationBraTs/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

myArray = []
fo = open("../corruptFilesList.txt", "r")
for line in fo:
    for nbr in line.split():
        myArray.append(int(nbr))
rangeStart = 10


#find file number in corrupted files list and delete file
for file in range(rangeStart, 70):
    fileNumber = '3' + str(file)
    try:
        if int(fileNumber) in myArray:
            dir = "BraTS20_Training_"+str(fileNumber)
            path = os.path.join(location, dir)
            shutil.rmtree(path)
            print ('folder '+str(fileNumber) +' has been deleted')

    except Exception as e:
        print(str(e))
        continue



