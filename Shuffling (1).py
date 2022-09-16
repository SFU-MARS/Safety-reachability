
# In this code I wanted to create new files which comes from portion (12) of elements from multiple (20)files

import os				# for file handling functions
import numpy as np		# for math functions
import pickle			# for reading/writing pickle files

# this is the directory where we find all of the pickle files we want to combine
data_directory = '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0914-new'

# gather a list of all files in that directory that end with the PKL extension
data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
#dictionaries = {}

# iterate through the files we discovered; add the contents to each to a master list of dictionaries
listOfDicts = []
for data_file in data_files:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        listOfDicts.append(data)

# verify that we have all of our entries
# print (np.size(listofdict))
train_size = 360
# [ADD]: you should explain why this is 10, and use a constant for it instead of a "magic number" that has no meaning to other users reading your code
for k in range(6):

  # initialize empty lists of our data
  posez = []
  imagez = []
  waypointActionz = []
  labelz = []

  # [ADD]: we've precomputed these indices because it's faster and easier to change, but maybe explain what they are; what are these 12s and why is this significant?
  index1 = 10 * k
  index2 = 10 * (k + 1)

  # [ADD]: you should explain why this is 20, and use a constant for it instead of a "magic number" that has no meaning to other users reading your code;
  #        if this 20 is supposed to represent the number of files, then you should iterate from 0 to range(len(data_files))---you don't assume a known number
  #        of files in your previous file-loading code, so you shouldn't assume a known number here, either
  for i in range(train_size):

    posez.append(listOfDicts[i]['start_pose'][index1: index2])
    imagez.append(listOfDicts[i]['image'][index1: index2])
    waypointActionz.append(listOfDicts[i]['waypointAction'][index1: index2])
    labelz.append(listOfDicts[i]['labels'][index1: index2])

  # I had to change the size to (240,3D) when originally they were array of (20, 12, 3D) , 20 is the number of files, and 12  length of slice from each of them
  dictionary = {'start_pose': np.array(posez).reshape((-1,2,1,1)),'image':np.array(imagez).reshape((-1,224,224,3)),'waypointAction':np.array(waypointActionz).reshape((-1,1,1,4)),'labels':np.array(labelz).reshape(((-1,1))) }
  # dictionary = {'start_pose': np.array(posez),
  #               'image': np.array(imagez),
  #               'waypointAction': np.array(waypointActionz),
  #               'labels': np.array(labelz)}
  # dictionaries.append(dictionary)

  # finally, output the master dictionary for this sample
  f = open("sample" + str(k) + '.pkl', "wb")
  pickle.dump(dictionary, f)
  f.close()
