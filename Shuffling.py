
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

data_directory= '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0729'
data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
databig={}
keys=[]
for data_file in data_files:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        listofdict.append(data)
each=1
m=int(60/each)              #60 muber of samles in each pkl file 
# print (len(listofdict))
for k1 in range(0,m):
# for k1 in range(1):

  posez = []
  imagez = []
  waypointActionz = []
  labelz = []
  train_size=50

  for i in range(train_size):
  # for i in range(2):
    # print(listofdict[i]['start_pose'][each*k1:each*(k1+1)])
    # print (k1)
    # print (i)
    #
    posez.append(np.squeeze(listofdict[i]['start_pose'][each*k1:each*(k1+1)]))
    imagez.append(np.squeeze(listofdict[i]['image'][each*k1:each*(k1+1)]))
    waypointActionz.append(np.squeeze(listofdict[i]['waypointAction'][each*k1:each*(k1+1)]))
    labelz.append(np.squeeze(listofdict[i]['labels'][each*k1:each*(k1+1)]))





  # import matplotlib.pyplot as plt
  #
  # plt.hist((np.array(labelz).reshape(1800, 1)))
  # plt.show()
  # posez_concat=np.concatenate(posez)
  # imagez_concat = np.concatenate(imagez)
  # waypointActionz_concat = np.concatenate(waypointActionz)
  # # labelz_concat = np.concatenate(labelz)
  # labelz_concat = labelz

  # posez_merged = np.expand_dims(posez_concat, axis=0)
  # imagez_merged = np.expand_dims(imagez_concat, axis=0)
  # waypointActionz_merged = np.expand_dims(waypointActionz_concat, axis=0)
  # labelz_merged = np.expand_dims(labelz_concat, axis=0)

 # I had to change the size to (240,3D) when originally they were array of (20, 12, 3D) , 20 is the number of files, and 12  length of slice from each of them
 #  dictionary = {'start_pose': np.array(posez).reshape((train_size*each,2,1,1)),'image':np.array(imagez).reshape((train_size*each,224,224,3)),'waypointAction':np.array(waypointActionz).reshape((train_size*each,1,1,4)),'labels':np.array(labelz).reshape(((train_size*each,1))) }
  dictionary = {'start_pose': np.stack(posez).reshape(2,1,1,50),
                'image': np.stack(imagez).reshape(224,224,3,50),
                'waypointAction': np.stack(waypointActionz).reshape(1,1,4,50),
                'labels': np.stack(labelz).reshape(1,50)}
  # dictionaries.append(dictionary)
  f  = open("sample"+str(1)+str(k1)+'.pkl', "wb")
  pickle.dump(dictionary, f)
  f.close()

for k in range(0,m):

  posez1 = []
  imagez1 = []
  waypointActionz1 = []
  labelz1 = []
  dictionaries1 = []

  for i in range(train_size,train_size+50,1):

    posez1.append(np.squeeze(listofdict[i]['start_pose'][each*k:each*(k+1)]))
    imagez1.append(np.squeeze(listofdict[i]['image'][each*k:each*(k+1)]))
    waypointActionz1.append(np.squeeze(listofdict[i]['waypointAction'][each*k:each*(k+1)]))
    labelz1.append(np.squeeze(listofdict[i]['labels'][each*k:each*(k+1)]))

# import matplotlib.pyplot as plt
#
# plt.hist((np.array(labelz1).reshape(600, 1)))
# plt.show()
#   posez_concat1 = np.concatenate(posez1)
#   imagez_concat1 = np.concatenate(imagez1)
#   waypointActionz_concat1 = np.concatenate(waypointActionz1)
#   labelz_concat1 = labelz1
#
#   posez_merged1 = np.expand_dims(posez_concat1, axis=0)
#   imagez_merged1 = np.expand_dims(imagez_concat1, axis=0)
#   waypointActionz_merged1 = np.expand_dims(waypointActionz_concat1, axis=0)
#   labelz_merged1 = np.expand_dims(labelz_concat1, axis=0)


# I had to change the size to (240,3D) when originally they were array of (20, 12, 3D) , 20 is the number of files, and 12  length of slice from each of them
# #   dictionary1 = {'start_pose': np.array(posez1).reshape((45*each,2,1,1)),'image':np.array(imagez1).reshape((45*each,224,224,3)),'waypointAction':np.array(waypointActionz1).reshape((45*each,1,1,4)),'labels':np.array(labelz1).reshape(((45*each,1))) }
#   dictionary1 = {'start_pose': posez_merged1,
#                 'image': imagez_merged1,
#                 'waypointAction': waypointActionz_merged1,
#                 'labels': labelz_merged1}
  dictionary1 = {'start_pose': np.stack(posez1).reshape(2,1,1,50),
                'image': np.stack(imagez1).reshape(224,224,3,50),
                'waypointAction': np.stack(waypointActionz1).reshape(1,1,4,50),
                'labels': np.stack(labelz1).reshape(1,1,50)}
  # dictionaries.append(dictionary)
  f = open("sample"+str(20)+str(k)+'.pkl', "wb")
  #200
  pickle.dump(dictionary1, f)
  f.close()

