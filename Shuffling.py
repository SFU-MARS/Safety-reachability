
# In this code I wanted to create new files which comes from portion (12) of elements from multiple (20)files

import os
import numpy as np
import pickle
listofdict=[]
posez=[]
imagez=[]
waypointActionz=[]
labelz=[]
dictionaries=[]

data_directory= '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/tmp6'
data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
databig={}
keys=[]
for data_file in data_files:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        listofdict.append(data)

# print (np.size(listofdict))
for k in range(10):

  posez = []
  imagez = []
  waypointActionz = []
  labelz = []

  for i in range(20):

    posez.append(listofdict[i]['start_pose'][12*k:12*(k+1)])
    imagez.append(listofdict[i]['image'][12*k:12*(k+1)])
    waypointActionz.append(listofdict[i]['waypointAction'][12*k:12*(k+1)])
    labelz.append(listofdict[i]['labels'][12*k:12*(k+1)])


 # I had to change the size to (240,3D) when originally they were array of (20, 12, 3D) , 20 is the number of files, and 12  length of slice from each of them
  dictionary = {'start_pose': np.array(posez).reshape((240,2,1,1)),'image':np.array(imagez).reshape((240,224,224,3)),'waypointAction':np.array(waypointActionz).reshape((240,1,1,4)),'labels':np.array(labelz).reshape(((240,1))) }
  # dictionary = {'start_pose': np.array(posez),
  #               'image': np.array(imagez),
  #               'waypointAction': np.array(waypointActionz),
  #               'labels': np.array(labelz)}
  # dictionaries.append(dictionary)
  f = open("sample"+str(k)+'.pkl', "wb")
  pickle.dump(dictionary, f)
  f.close()

