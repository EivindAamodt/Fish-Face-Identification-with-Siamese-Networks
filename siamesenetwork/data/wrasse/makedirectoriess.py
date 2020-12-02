import os
import shutil

for f in os.listdir("../../../wrasse/body"):  # body or face
    folderName = f.split("_")[0]    #treat right and left facing as same class
    # folderName = f.split("_")[0] + "_" + f.split("_")[1]  #splits between right and left facing
    if not os.path.exists(folderName):
        os.mkdir(folderName)
        shutil.copy(os.path.join('../../../wrasse/body', f), folderName)  # body or face
    else:
        shutil.copy(os.path.join('../../../wrasse/body', f), folderName)  # body or face

