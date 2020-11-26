import csv
import glob

import os
from PIL import Image


#####   removing the corrupted ones turns    #####
#####        training: xxxx -> 9645          #####
#####       validation: xxxx -> 3384         #####

imageFolder = "training/"
listImages = os.listdir(imageFolder)

corruptimagestraining = []

for img in listImages:
    imgPath = os.path.join(imageFolder, img)

    try:
        img = Image.open(imgPath)
        exif_data = img._getexif()
    except ValueError as err:
        print(err)
        print("Error on image: ", imgPath)
        a = imgPath.split("/")
        corruptimagestraining.append(a[1])

print(corruptimagestraining)


imageFolder = "validation/"
listImages = os.listdir(imageFolder)

corruptimagesvalidation = []

for img in listImages:
    imgPath = os.path.join(imageFolder, img)

    try:
        img = Image.open(imgPath)
        exif_data = img._getexif()
    except ValueError as err:
        print(err)
        print("Error on image: ", imgPath)
        a = imgPath.split("/")
        corruptimagesvalidation.append(a[1])

print(corruptimagesvalidation)


images = glob.glob("training/*.jpg")


with open('training.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # x = images[3000].split("_")
    for x in images:
        a = str(x.split(".")[0]+".jpg").split("\\")
        if a[1] not in corruptimagestraining:
            if x[10] != "_":
                asdd = x[9]+x[10]
                x = x.split("_")
                asd = str(x[0][9])+str(x[0][10]) + "_" + str(x[1].split(".")[0]+".jpg")
            else:
                asdd = x[9]
                x = x.split("_")
                asd = str(x[0][9]) + "_" + str(x[1].split(".")[0]+".jpg")
            print(asd, asdd)
            writer.writerow([asd, asdd])






images = glob.glob("validation/*.jpg")

with open('validation.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # x = images[3000].split("_")
    for x in images:
        a = str(x.split(".")[0] + ".jpg").split("\\")
        if a[1] not in corruptimagesvalidation:
            if x[12] != "_":
                asdd = x[11]+x[12]
                x = x.split("_")
                asd = str(x[0][11])+str(x[0][12]) + "_" + str(x[1].split(".")[0]+".jpg")
            else:
                asdd = x[11]
                x = x.split("_")
                asd = str(x[0][11]) + "_" + str(x[1].split(".")[0]+".jpg")
            print(asd, asdd)
            writer.writerow([asd, asdd])


