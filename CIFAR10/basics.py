#!/usr/bin/env python


import numpy as np
import math
import time
import os
import copy
from keras import backend as K

import sys
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from keras import backend as K
from scipy.stats import truncnorm, norm

################################################################
#
#  initialise HTS-based direct manipulations
#
################################################################

imageEnlargeProportion = 2
maxNumOfPointPerKeyPoint = 0 
    
# should be local_featureDims, but make it 1 
local_featureDims = 1

MCTS_all_maximal_time = 1
MCTS_level_maximal_time = 1
MCTS_multi_samples = 1

span = 255/float(255)
numSpan = 1
controlledSearch = ("L1",60)

featureDims = 3

def applyManipulation(image,span,numSpan):

    image1 = copy.deepcopy(image)

    #print('span.key:', span.keys())

    for elt in span.keys(): 
        if len(elt) == 2: 
            (fst,snd) = elt 
            if 1 - image[fst][snd] < image[fst][snd] : image1[fst][snd] -= numSpan[elt] * span[elt]
            else: image1[fst][snd] += numSpan[elt] * span[elt]
            if image1[fst][snd] < 0: image1[fst][snd] = 0
            elif image1[fst][snd] > 1: image1[fst][snd] = 1
        elif len(elt) == 3: 
            (fst,snd,thd) = elt 
            if 1 - image[fst][snd][thd] < image[fst][snd][thd] : image1[fst][snd][thd] -= numSpan[elt] * span[elt]
            else: image1[fst][snd][thd] += numSpan[elt] * span[elt]
            if image1[fst][snd][thd] < 0: image1[fst][snd][thd] = 0
            elif image1[fst][snd][thd] > 1: image1[fst][snd][thd] = 1
    return image1

def getResponse(kp):

    return kp.response
    
def siftApplyManipulationDirect(model, image):

    image1 = copy.deepcopy(image)

    if np.max(image1) <= 1: 
        image1 = (image1*255).astype(np.uint8)
    else: 
        image1 = image1.astype(np.uint8)

    image2 = copy.deepcopy(image)

    ls = []    
    kp_tmp = SIFT_Filtered_twoPlayer(image1)

    print("%s keypoints are found."%(len(kp_tmp)))
    for i in range(len(kp_tmp)):
    	  print("keypoint %s : %s, %s size: %s resp: %s"%(i, kp_tmp[i].pt[0], kp_tmp[i].pt[1], kp_tmp[i].size, kp_tmp[i].response))
    	  #(a, b) = (int(kp_tmp[i].pt[0]), int(kp_tmp[i].pt[1]))
    	   	  
    	  if (int(kp_tmp[i].pt[0]), int(kp_tmp[i].pt[1])) not in ls:
    	      ls.append((int(kp_tmp[i].pt[0]), int(kp_tmp[i].pt[1]), 0))
    	      ls.append((int(kp_tmp[i].pt[0]), int(kp_tmp[i].pt[1]), 1))
    	      ls.append((int(kp_tmp[i].pt[0]), int(kp_tmp[i].pt[1]), 2))

    #print('ls:', ls)

    for x, y, z in ls:
        #print('x, y:', x, y)
        if image2[x][y][z] > 0.5:
            image2[x][y][z] = 0
        else:
            image2[x][y][z] = 1

    return image2

        
def SIFT_Filtered_twoPlayer(image): #threshold=0.0):

    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(image,None)
    
    return  kp
            
    
def SIFT_Filtered(image, numOfPoints): #threshold=0.0):

    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    return  getPoints(image, kp, numOfPoints)

def getPoints(image, kp, n): 
    dist = getDistribution(image, kp)
    indices = [] #np.zeros(dist.shape)
    for i in range(len(dist)):
        for j in range(len(dist[0])): 
            indices.append(i*len(dist)+j) # [i][j] = (i,j)
    l =  np.random.choice(indices, n, p = dist.flatten())
    l2 = []
    for ind in l: 
        l2.append(getPixelLoc(ind,image))
    return dist, list(set(l2))

def getPixelLoc(ind, image):
    return (ind/len(image), ind%len(image))

######################################################################


def diffImage(image1,image2):
    return list(zip (*np.nonzero(np.subtract(image1,image2))))    #MY_MOD
    
    
def diffPercent(image1,image2): 
        return len(diffImage(image1,image2)) / float(image1.size)
        
def numDiffs(image1,image2): 
        return len(diffImage(image1,image2))
    

    
def euclideanDistance(image1,image2):
    return math.sqrt(np.sum(np.square(np.subtract(image1,image2))))
    
def l1Distance(image1,image2):
    return np.sum(np.absolute(np.subtract(image1,image2)))

def l0Distance(image1,image2):
    return np.count_nonzero(np.absolute(np.subtract(image1,image2)))

    
def mergeTwoDicts(x,y):
    z = x.copy()
    z.update(y)
    return z


def getDistribution(image, kp):

    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import multivariate_normal
    import scipy.stats
    import numpy.linalg
    
    dist = np.zeros(image.shape[:2])
    i = 1
    for  k in kp: 
        #print(i)
        i += 1
        a = np.array((k.pt[0],k.pt[1]))
        for i in range(len(dist)): 
            for j in range(len(dist[0])): 
                b = np.array((i,j))
                dist2 = numpy.linalg.norm(a - b)
                dist[i][j] += scipy.stats.norm.pdf(dist2, loc=0.0, scale=k.size) * k.response
                    
    return dist / np.sum(dist)
    
############################################################
#
#  auxiliary functions
#
################################################################
    
def getWeight(wv,bv,layerIndex):
    wv = [ (a,(p,c),w) for (a,(p,c),w) in wv if p == layerIndex ]
    bv = [ (p,c,w) for (p,c,w) in bv if p == layerIndex ]
    return (wv,bv)
    
def numberOfFilters(wv):
    return np.amax(list(zip (*(list(zip (*wv))[1])  ))[1])

#  the features of the last layer
def numberOfFeatures(wv):
    return np.amax(list(zip (*(list(zip (*wv))[0])  ))[1])
    
def otherPixels(image, ps):
    ops = []
    if len(image.shape) == 2: 
          for i in range(len(image)): 
              for j in range(len(image[0])): 
                  if (i,j) not in ps: ops.append((i,j))
    return ops


def nprint(str):
    return      
        
    
def saveImg(layer,image,filename):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    import matplotlib
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pyplot as plt
    
    plt.imsave(filename, image * 255, format="png", cmap=matplotlib.cm.Greys)    


def saveImgCifar(layer, image, filename):
    """
    """
    import cv2
    import copy

    image_cv = copy.deepcopy(image)
    
    if np.max(image_cv) <= 1: 
        image_cv1 = (image_cv*255).astype(np.uint8)
    else: 
        image_cv1 = image_cv.astype(np.uint8)
    

    params = list()
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)
    
    cv2.imwrite(filename, image_cv1, params)



############################################################
#
#  initialise a region for the input (only for searchMCTS)
#
################################################################   


repeatedManipulation = "disallowed"

 
def initialiseRegionActivation(model,manipulated,image): 
    
    #print len(manipulated)

    layerType = "InputLayer"

    if layerType == "InputLayer":
        nextSpan = {}
        nextNumSpan = {}
        # print('image.size:', image.size, 'manipulated:', manipulated, 'len(image.shape):', len(image.shape))
        if len(image.shape) == 3: 
            # decide how many elements in the input will be considered
            if image.size < featureDims : 
                numDimsToMani = image.size 
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            #ls = getTop2DActivation(image,manipulated,[],numDimsToMani,-1)
            ls = getTop3DActivation(image,manipulated,[],numDimsToMani,-1)
            #ls = getTopActivation(image,manipulated,-1,numDimsToMani)

        for i in ls: 
            nextSpan[i] = span
            nextNumSpan[i] = numSpan

    else: 
        print ("initialiseRegionActivation: Unknown layer type ... ")

    # print "nextSpan: %s nextNumSpan: %s numDimsToMani: %s"%(nextSpan,nextNumSpan,numDimsToMani)
    return (nextSpan,nextNumSpan,numDimsToMani)
    
    

    
############################################################
#
#  auxiliary functions (only for searchMCTS)
#
################################################################
    
# This function only suitable for the input as a list, not a multi-dimensional array

def getTopActivation(image,manipulated,layerToConsider,numDimsToMani): 

    avoid = repeatedManipulation == "disallowed"
    
    #avg = np.sum(image)/float(len(image))
    #nimage = list(map(lambda x: abs(avg - x),image))
    avg = np.average(image)
    nimage = np.absolute(image - avg) 

    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        print('nimage ', i, ' :', nimage[i])
        if len(topImage) < numDimsToMani and ((not avoid) or (i not in manipulated)):
            print('topImage ', i) 
            topImage[i] = nimage[i]
        else: 
            bl = False
            for k, v in topImage.items():
                print('k, v:', k, v)
                #if v < nimage[i] and not (k in toBeDeleted) and ((not avoid) or (i not in manipulated)):
                if (v-nimage[i]).any() and not (k in toBeDeleted) and ((not avoid) or (i not in manipulated)):
                        toBeDeleted.append(k)
                        bl = True
                        break
            if bl == True: 
                topImage[i] = nimage[i]
    for k in toBeDeleted: 
        del topImage[k]
    return topImage.keys()

def getTop2DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed"
            
    avg = np.average(image)
    nimage = np.absolute(image - avg) 


    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if len(topImage) < numDimsToMani and ((not avoid) or ((i,j) not in manipulated)): 
                topImage[(i,j)] = nimage[i][j]
            else: 
                bl = False 
                for (k1,k2), v in topImage.items():
                    if v < nimage[i][j] and not ((k1,k2) in toBeDeleted) and ((not avoid) or ((i,j) not in manipulated)):  
                        toBeDeleted.append((k1,k2))
                        bl = True
                        break
                if bl == True: 
                    topImage[(i,j)] = nimage[i][j]

    for (k1,k2) in toBeDeleted: 
        del topImage[(k1,k2)]


    return topImage.keys()


def getTop3DActivation(image,manipulated,ps,numDimsToMani,layerToConsider): 

    avoid = repeatedManipulation == "disallowed"
    
    avg = np.average(image)
    nimage = np.absolute(image - avg) 
                
    # do not care about the first dimension
    # only care about individual convolutional node
    if len(list(ps)) > 0:            #MY_MOD
        if len(list(ps)[0]) == 3:      #MY_MOD
            (p1,p2,p3) = zip(*ps)
            ps = zip(p2,p3)
    ks = []
    pointsToConsider = []
    for i in range(numDimsToMani): 
        #if i <= (len(list(ps)) - 1):   #MY_MOD
        if len(list(ps)) - 1 >= i and len(list(ps)) > 0:  #MY_MOD
            #tmp_ps = list(ps)
            print('TTT', i, list(ps), len(list(ps)) - 1)
            (x,y) = list(ps)[i]   #MY_MOD
            nps = [ (x-x1,y-y1) for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >=0 ]
            pointsToConsider = pointsToConsider + nps
            ks = ks + list(findFromArea3D(image,manipulated,avoid,nimage,nps,1,ks))     #MY_MOD
        else: 
            ks = ks + list(findFromArea3D(image,manipulated,avoid,nimage,pointsToConsider,1,ks))
    return ks
    

def findFromArea3D(image,manipulated,avoid,nimage,ps,numDimsToMani,ks):
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                if len(topImage) < numDimsToMani and ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    topImage[(i,j,k)] = nimage[i][j][k]
                elif ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks: 
                    bl = False 
                    for (k1,k2,k3), v in topImage.items():
                        if v < nimage[i][j][k] and not ((k1,k2,k3) in toBeDeleted) and ((not avoid) or ((i,j,k) not in manipulated)):  
                            toBeDeleted.append((k1,k2,k3))
                            bl = True
                            break
                    if bl == True: 
                        topImage[(i,j,k)] = nimage[i][j][k]
    for (k1,k2,k3) in toBeDeleted: 
        del topImage[(k1,k2,k3)]
    return topImage.keys()
    

def initialiseRegions(model,image,manipulated):
    allRegions = []
    num = image.size/featureDims
    newManipulated1 = []
    newManipulated2 = manipulated
    while num > 0 : 
        oneRegion = initialiseRegionActivation(model,newManipulated2,image)
        allRegions.append(oneRegion)
        newManipulated1 = copy.deepcopy(newManipulated2)
        #print('newMani2:', newManipulated2, 'oneRegion[0].key:', oneRegion[0].keys())
        newManipulated2 = list(set(newManipulated2 + list(oneRegion[0].keys())))
        #print('newMani:', newManipulated2)
        if newManipulated1 == newManipulated2: break
        num -= 1
    return allRegions
