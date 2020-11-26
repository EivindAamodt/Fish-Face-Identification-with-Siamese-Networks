import os, shutil


for f in os.listdir("body"):
    folderName1 = f.split("_")[0]# + "_" + f.split("_")[1]
    if not os.path.exists(folderName1):
        os.mkdir(folderName1)
        shutil.copy(os.path.join('body', f), folderName1)
    else:
        shutil.copy(os.path.join('body', f), folderName1)