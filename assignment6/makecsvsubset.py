import random
fid = open("training.csv", "r")
li = fid.readlines()
fid.close()
print(li)

random.shuffle(li)
print(li)

fid = open("subsettraining.txt", "w")
for i in range(1000):
    fid.writelines(li[i])
fid.close()


fid = open("validation.csv", "r")
li = fid.readlines()
fid.close()
print(li)

random.shuffle(li)
print(li)

fid = open("subsetvalidation.txt", "w")
for i in range(500):
    fid.writelines(li[i])
fid.close()