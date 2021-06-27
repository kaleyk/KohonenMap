import sys
import numpy as np
import matplotlib.pyplot as cm
import matplotlib.pyplot as plt
import random
import math
from random import randrange

'''
application of Kohonen Network on random colours
I create a random colour map, then proceed to train it upon user specified amount of epochs, 
learning rate, and file prefix.
'''

#function to generate colour vectors, takes n as number of colours to be generated for clustering
def Gen_Colours(n):
    colours =  []
    for x in range(n):
        colours.append([random.random(),random.random(),random.random()])
    return colours

#function to create a map of colour vectors
def Gen_Image(colors, x, y):
    img = []
    for row in range(x):
        for col in range(y):
            img.append(colors[randrange(len(colors))])
    img  = np.asarray(img, dtype=np.float32)
    img = img.reshape(x, y, 3)
    return img

#Func to calculate the euclidean distance between vectors
def Euclidean_Dist(V, t): 
    distance = np.sum(np.square(V-t))
    return distance

#Func to find the best matching unit
def findBMU(init_image,map,RandX,RandY,Ox,Oy):
    #randomize centroid x and y values
    centroid = random.randint(0,len(map)),random.randint(0,len(map))
    temp = math.inf

    #returns centroid considered to be bmu
    for x in range(len(map)):
        for y in range(len(map)):
            distance = Euclidean_Dist(map[x][y],init_image[x][y])
            if distance > temp:
                temp = distance
                centroid = [x,y]

    return centroid

#input vector, initial learning rate, m = map, im = img, x and y are dimensions, fname is specified by user
def Iterate(input, lr, m, im, x , y, fname):
     #formatting of saves
    counter = 1
    img_num = 1

    for S in range(input):
        #choose random input vector (x,y)
        RandX = random.randint(0,len(m)-1)
        RandY = random.randint(0,len(m)-1)

        #learning rate decay
        CONST = (1.0 - ((S*1.0)/input))
        
        #find best matching unit
        bmu = findBMU(im, m, RandX, RandY, len(m), len(m))

        #modification of learning rate
        emp = int(CONST * max_range)
        learningrate = CONST * lr

        #modify vector positions within neighbourhood using euclidean distance and bmu
        for i in range(x):
            for j in range(y):
                distance = Euclidean_Dist(np.array(bmu),np.array([i,j])) 
                if distance < emp:
                    m[i][j] = m[i][j] + learningrate * (init_image[RandX][RandY] - m[i][j])

        file_name = "{}{}".format(fname, img_num)

        #display new map after each epoch
        #save the img every 50 epochs
        if counter == 50: 
            #counter is reset to 1 after 50 epochs is hit
            counter = 1
            img.set_data((m*255).astype(np.uint8))
            cm.pause(.01)
            cm.draw()
            cm.savefig(file_name)
            img_num = img_num + 1
        else:
            #if not 50th epoch proceed as normal
            img.set_data((m*255).astype(np.uint8))
            cm.pause(.01)
            cm.draw()
        
        counter = counter + 1

#dimensions of map specified here
x = 200
y = 200
max_range = x + y 

#create initial colour map
init_image = Gen_Image(Gen_Colours(5), x, y)
img = cm.imshow((init_image*255).astype(np.uint8))
map = np.random.rand(x,y,3)

#USER INPUT FIELDS FOR EPOCHS, LEARNING RATE, AND FILE PREFIX
#user specifies number of epochs to be run
num_epoch = int(input("Please Enter # of Epoch's (recommended 200-1000): "))

#User specifies Learning rate
lr = float(input("Please Enter Learning rate (recommended 0.1-0.9): "))

#User specifies file prefix
fn = str(input("Please Enter String for file prefix: "))

#Algorithm iterates here
Iterate(num_epoch, lr, map, init_image, x, y, fn)

#display final plt/image
plt.imshow((map*255).astype(np.uint8))
plt.show()

sys.exit()