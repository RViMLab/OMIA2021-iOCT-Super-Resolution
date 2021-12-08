import matplotlib.pyplot as plt
import numpy as np



def showManyImages(imageList):

    i=0
    j=0
    cols=10
    rows=int(np.array(imageList).shape[0]/cols)
    print("rows:",rows)
    print("cols:", cols)
    fig, axes = plt.subplots(rows, cols)

    for counter in range(rows*cols):
        print("i",i)
        print("j",j)
        if(i==cols):
            i=0
            j=j+1
        if(j==rows):
            break

        axes[j][i].imshow(imageList[counter],cmap='gray')
        i = i + 1
    plt.show()