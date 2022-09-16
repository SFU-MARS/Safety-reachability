
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
dictionary=[]
dictionary1=[]
dict_list=[]
num_sample_each_file=[]

data_directory= '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0914-new'
data_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
databig={}
keys=[]
count1=0

for data_file in data_files:
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        num_sample_each_file.append(data['waypointAction'].shape[0])
        if data['waypointAction'].shape[0]==60:
            listofdict.append(data)

from_each = 10
m = int(max(num_sample_each_file) / from_each)
#            #60 number of samles in each pkl file
# # print (len(listofdict))
# for k1 in range(m):
# # for k1 in range(1):
train_size = 360
# my_List = [165, 166, 285, 292, 413, 452]


for k in range(0, m):

    for i in range(1, train_size):
        # for i in my_List:
        # print(listofdict[i]['start_pose'][each*k1:each*(k1+1)])
        # print (k1)
        # print (i)
        #

        posez.append(np.squeeze(listofdict[i]['start_pose'][from_each * k:from_each * (k + 1)]))
        imagez.append(np.squeeze(listofdict[i]['image'][from_each * k:from_each * (k + 1)]))
        # waypointActionz.append(np.squeeze(listofdict[i]['waypointAction'][:]))
        waypointActionz.append(np.squeeze(listofdict[i]['waypointAction'][from_each * k:from_each * (k + 1)]))
        labelz.append(np.squeeze(listofdict[i]['labels'][from_each * k:from_each * (k + 1)]))
        if 1 in labelz[0]:
            count1 =+1

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
        #  dictionary = {'start_pose': np.stack(posez).reshape(2,1,1,50),
        #                'image': np.stack(imagez).reshape(224,224,3,50),
        #                'waypointAction': np.stack(waypointActionz).reshape(1,1,4,50),
        #                'labels': np.stack(labelz).reshape(1,50)}

        # dictionary = {'start_pose': np.stack(posez).reshape(2, 1, -1),  # (2, 1, 60)
        #           'image': np.stack(imagez).reshape(224, 224, 3, -1),
        #           'waypointAction': np.stack(waypointActionz).reshape(4, 1, -1),
        #           'labels': np.stack(labelz).reshape(1, -1)}

    dictionary = {'start_pose': np.stack(posez).reshape(-1, 2, 1),  # (2, 1, 60)
                  'image': np.stack(imagez).reshape(-1, 224, 224, 3),
                  'waypointAction': np.stack(waypointActionz).reshape(-1, 4, 1),
                  'labels': np.stack(labelz).reshape(-1, 1)}

        # if (i % 10) != 0:

        # dict_list.append(dictionary)

    posez = []
    imagez = []
    waypointActionz = []
    labelz = []

    # if ((i+1) % 10) == 0:

        # dictionaries.append(dictionary)
    f = open("sample" + str(1) + str(int(k)) + '.pkl', "wb")
    pickle.dump(dictionary, f)
    dict_list = []


    f.close()

posez1 = []
imagez1 = []
waypointActionz1 = []
labelz1 = []
dict_list1 = []

for k in range(0, m):

    for i in range(train_size + 1, train_size + 120, 1):

        posez1.append(np.squeeze(listofdict[i]['start_pose'][from_each * k:from_each * (k + 1)]))
        imagez1.append(np.squeeze(listofdict[i]['image'][from_each * k:from_each * (k + 1)]))
        waypointActionz1.append(np.squeeze(listofdict[i]['waypointAction'][from_each * k:from_each * (k + 1)]))
        labelz1.append(np.squeeze(listofdict[i]['labels'][from_each * k:from_each * (k + 1)]))
        if 1 in labelz1[0]:
            count1 =+1

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
        #   dictionary1 = {'start_pose': np.stack(posez1).reshape(2,1,1,50),
        #                 'image': np.stack(imagez1).reshape(224,224,3,50),
        #                 'waypointAction': np.stack(waypointActionz1).reshape(1,1,4,50),
        #                 'labels': np.stack(labelz1).reshape(1,1,50)}

    dictionary1 = {'start_pose': np.stack(posez1).reshape(-1, 2, 1),
                       'image': np.stack(imagez1).reshape(-1, 224, 224, 3),
                       'waypointAction': np.stack(waypointActionz1).reshape(-1, 4, 1),
                       'labels': np.stack(labelz1).reshape(-1, 1)}

        # if i % 10 != 0:
        # dict_list1.append(dictionary1)

    posez1 = []
    imagez1 = []
    waypointActionz1 = []
    labelz1 = []

# if (i+1) % 10 == 0:
    # dictionaries.append(dictionary)
    f = open("sample" + str(2) + str(int(k) ) + '.pkl', "wb")
    # 200
    pickle.dump(dictionary1, f)
    dict_list1 = []

    f.close()
