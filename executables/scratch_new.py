
def generate_QandActionforImage(image):
    result = list()
    for an_action in generated_actions():
        next_state=apply_action(an_action, current_state)
        V=calculate_value(next_state)
        Q=calculate_quality(V)
        result.append([an_action,Q])

for an_image in images:
    generate_QandActionforImage(an_image)


---------------------------------------------------------------------------------
#walking
def walking(current_state):
    waypoint=list()
    for 1sec:
    while not collide:
        image=take_picture()
        action=best_find_action(image, current_state)
        current_state=apply_action(action)
        waypoint.append([current_state])
        return



result=list()
for an_image in images:
    for an_action in generated_actions():
        next_state=apply_action(an_action, current_state)
        V=calculate_value(next_state)
        Q=calculate_quality(V)
        result.append([an_image,an_action,Q])


----------------------------------------------------------------------------------

#training a simulator

def generate_training_dataset(num_data=10):
    # input
    # output
    dataset=list()
    dataset_w = list()
    #num_data=10
    for sim_number in range (num_data):
        s=simulator()
        image=s.state.render()
        for an_action in generate_actions():
            next_state=apply_action(s.state,an_action)
            V = calculate_value(next_state)
            Q = calculate_Q(V)
            dataset.append([sim_number,image, an_action, Q])
            waypoint=walking()
            dataset_w.append(([sim_number,image, waypoint]))
    return dataset , dataset_w

def calculate_Q(state,class):
def calculate_waypoint():

def best_find_action(image,dataset):
    calculate_Q()
    for item in dataset
        img=item[1]
        if img==image
            #max_Q=max(item[3] for item in dataset)
            m=maxIndex()
            best_action= dataset[item[2][m[0]]]

def maxIndex(nestedlist):
    m = (0, 0)
    for i in range(len(nestedlist)):
        for j in range (len(nestedlist(i))):
            if nestedlist[i][j]>nestedlist[m[0]][m[1]]:
                m=(i,j)

    return m



        while not collide:
            image=take_picture()
            action=best_guess_action(image, current_state)
            current_state=apply_action(action)


def filter_dataset(dataset):
    dataset2=list()
    for item in dataset
        Q = item[3]
    if rand> 1/2 && Q<100
        dataset2.append([sim_number,image, an_action, Q])
        if rand < 1 / 2 & & Q > 100
        dataset2.append([sim_number,image, an_action, Q])

    return dataset2
-----------------------------------
#train(dataset2)