# =================================================  Settings  =================================================

path = "C:\\Users\\User\\"
load = "Train.csv"
saveAs = 'Augmented data.csv'
extra_images = 1000             ## per Number.
start_after_cutoff = 0          ## in case we will split the dataset for testing and training.
show_problematic_creations = 1  ## show 10 problematic created numbers that got discarded.

# =================================================     End    =================================================


import numpy as np
import pandas as pd
from matplotlib import pyplot

def reshape_image(arr, num):
    arr1 = np.roll(arr, -num, axis=1)  # left
    arr2 = np.roll(arr, num, axis=1)   # right
    arr3 = np.roll(arr, -num, axis=0)  # up
    arr4 = np.roll(arr, num, axis=0)   # down
    arr5 = np.roll(arr1, -num, axis=0) # LU
    arr6 = np.roll(arr2, -num, axis=0) # RU
    arr7 = np.roll(arr1, num, axis=0)  # LD
    arr8 = np.roll(arr2, num, axis=0)  # RD
    arr9 = np.roll(arr, 2, axis=1)     # 2xright
    arr10 = np.roll(arr, -2, axis=1)   # 2xleft
    arr11 = np.roll(arr, -2, axis=0)   # 2xup
    arr12 = np.roll(arr, 2, axis=0)    # 2xdown
    arr13 = np.roll(arr9, 2, axis=0)   # 2xRD
    arr14 = np.roll(arr9, -2, axis=0)  # 2xRU
    arr15 = np.roll(arr10, -2, axis=0) # 2xLU
    arr16 = np.roll(arr10, 2, axis=0)  # 2xLD
    
    return arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10, arr11, arr12, arr13, arr14, arr15, arr16


dataset = np.array(pd.read_csv(path+load))
list_of_new_images = []
counter = {}
row, col = 1, 1
for i in range(0, 10):
    counter[i] = extra_images

dataset_last_row = dataset.shape[0]
list_of_images = dataset[0:dataset_last_row].T
labels = list_of_images[0]
list_of_images = list_of_images[1:]
problematic_numbers_shown = 0


for index in range(start_after_cutoff, dataset_last_row):
    print(index+1)

    a = labels[index]
    if counter[a] < 0:
        continue
    
    main_image = list_of_images[:, index, None].reshape((28, 28))
    for image in reshape_image(main_image, 1):

        ## Check if image overfloaded
        dic = {"U":0, "D":0, "L":0, "R":0}      
        for x in range(28):
            if image[x][27] > 0:
                dic["R"] += 1
            if image[x][0] > 0:
                dic["L"] += 1
            if image[27][x] > 0:
                dic["D"] += 1
            if image[0][x] > 0:
                dic["U"] += 1
        
        ## If the image has pixels at both opposite sides, dont include it in the new pictures
        if not (dic["R"] and dic["L"]) and not (dic["U"] and dic["D"]):
            counter[a] -= 1
            list_of_new_images.append(np.concatenate(([labels[index]], image.flatten())) )
        elif show_problematic_creations and problematic_numbers_shown < 10:
            pyplot.imshow(image)
            pyplot.show()
            problematic_numbers_shown += 1

pd.DataFrame(list_of_new_images).to_csv(path+saveAs, index = False, header= False)
print("----> Saved <----")