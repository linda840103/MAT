#!/usr/bin/env python

"""
HTS

"""

import sys
import time
import numpy as np
import copy 
import random
import math
import ast
import stopit
import matplotlib.pyplot as plt
import matplotlib as mpl
import keras
import tensorflow
from scipy import ndimage
from keras import backend as K

from basics import *
import cifar_nn as NN

from searchMCTS import searchMCTS


    
def LABELS(index):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
    return labels[index]

def sift_apply_direct(model, image):
	
    re = False	  
    start_time = time.time()	  
    print ("directly apply manupulation on the image ... ")
    (originalClass,originalConfident) = NN.predictImage(model,image)
    #print('originalClass: ', originalClass, 'originalConfident: ', originalConfident)

    image1 = siftApplyManipulationDirect(model, image)

    (newClass,newConfident) = NN.predictImage(model,image1)
    
    eudist = euclideanDistance(image,image1)
    l1dist = l1Distance(image,image1)
    l0dist = l0Distance(image,image1)
    percent = diffPercent(image,image1)
    print("euclidean distance %s"%(eudist))
    print("L1 distance %s"%(l1dist))
    print("L0 distance %s"%(l0dist))
    print("manipulated percentage distance %s"%(percent))
    print("class is changed into %s with confidence %s\n"%(newClass, newConfident))

    originalClassStr = LABELS(int(originalClass))
    newClassStr = LABELS(int(newClass))

    path0 = "%s/%s_org_as_%s_withConf_%s.png"%("image", originalClassStr, originalClassStr, originalConfident)
    path1 = "%s/%s_org_as_%s_withConf_%s.png"%("image", originalClassStr, newClassStr, newConfident)
    #saveImgCifar(-1, image, path0)
    #saveImgCifar(-1, image1, path1)

    return image1
    
def sift_mcts_single(model, image):

    re = False
    
    # 60, 15, 3
    MCTS_all_maximal_time = 1
    MCTS_level_maximal_time = 1
    MCTS_multi_samples = 1
    
    span = 255/float(255)
    numSpan = 1
    #errorBounds = {}
    #errorBounds[-1] = 1.0
    
    start_time = time.time()
    
    print ("directly handling the image ... single")
          
    (originalClass,originalConfident) = NN.predictImage(model,image)

    # initialise a search tree
    st = searchMCTS(model,image,-1)
    st.initialiseActions()
    
    if st.actions == {} :
         return (image, False, originalClass, originalClass, 0, 0, 0, 0, 0)

    start_time_all = time.time()
    runningTime_all = 0
    numberOfMoves = 0

    print("best possible one is %s"%(str(st.bestCase)))
    while st.terminalNode(st.rootIndex) == False and st.terminatedByControlledSearch(st.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time: 

         print("the number of moves we have made up to now: %s"%(numberOfMoves))
         eudist = st.euclideanDist(st.rootIndex)
         l1dist = st.l1Dist(st.rootIndex)
         l0dist = st.l0Dist(st.rootIndex)
         percent = st.diffPercent(st.rootIndex)
         diffs = st.diffImage(st.rootIndex)
         print("euclidean distance %s"%(eudist))
         print("L1 distance %s"%(l1dist))
         print("L0 distance %s"%(l0dist))
         print("manipulated percentage distance %s"%(percent))
         print("manipulated dimensions %s"%(diffs))
         
         start_time_level = time.time()
         runningTime_level = 0
         childTerminated = False
         
         while runningTime_level <= MCTS_level_maximal_time:
             (leafNode,availableActions) = st.treeTraversal(st.rootIndex)
             newNodes = st.initialiseExplorationNode(leafNode,availableActions)
             for node in newNodes:
                 (childTerminated, value) = st.sampling(node,availableActions)
                 if childTerminated == True: break
                 st.backPropagation(node,value)
             if childTerminated == True: break
             runningTime_level = time.time() - start_time_level
             print("best possible one is %s"%(st.showBestCase()))
         bestChild = st.bestChild(st.rootIndex)
         st.makeOneMove(bestChild)

         image1 = applyManipulation(st.image,st.spans[st.rootIndex],st.numSpans[st.rootIndex])
         diffs = st.diffImage(st.rootIndex)

         (newClass,newConfident) = NN.predictImage(model,image1)
         print("new class %s confidence: %s"%(newClass, newConfident))
         
         if newClass != originalClass: break

         if childTerminated == True: break

         # store the current best
         (_,bestSpans,bestNumSpans) = st.bestCase
         image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
         
         runningTime_all = time.time() - runningTime_all
         numberOfMoves += 1

    (_,bestSpans,bestNumSpans) = st.bestCase
    image1 = applyManipulation(st.image,bestSpans,bestNumSpans)
    (newClass,newConfident) = NN.predictImage(model,image1)
    print("best new class %s confidence: %s"%(newClass, newConfident))
    
    re = newClass != originalClass

    if re == True:     

         print("\nfound an adversary image within prespecified bounded computational resource. The following is its information: ")
         print("difference between images: %s"%(diffImage(image,image1)))
    
         eudist = euclideanDistance(st.image,image1)
         l1dist = l1Distance(st.image,image1)
         l0dist = l0Distance(st.image,image1)
         percent = diffPercent(st.image,image1)
         print("euclidean distance %s"%(eudist))
         print("L1 distance %s"%(l1dist))
         print("L0 distance %s"%(l0dist))
         print("manipulated percentage distance %s"%(percent))
         print("class is changed into %s with confidence %s\n"%(newClass, newConfident))

         originalClassStr = LABELS(int(originalClass))
         newClassStr = LABELS(int(newClass))

         path0 = "%s/%s_org_as_%s_withConf_%s_EU%s_L1%s_L0%s_per%s.png"%("image", originalClassStr, originalClassStr, originalConfident, eudist, l1dist, l0dist, percent)
         path1 = "%s/%s_org_as_%s_withConf_%s_EU%s_L1%s_L0%s_per%s.png"%("image", originalClassStr, newClassStr, newConfident, eudist, l1dist, l0dist, percent)
         saveImgCifar(-1, image, path0)
         saveImgCifar(-1, image1, path1)

         del st
         return (image1, True, originalClass, newClass, newConfident, eudist, l1dist, l0dist, percent)
    else:
         
         print("\nfailed to find an adversary image within prespecified bounded computational resource. ")
         del st
         return (sift_apply_direct(model, image), False, originalClass, originalClass, 0, 0, 0, 0, 0)

    

def reportInfo(image,wk):

    # exit only when we find an adversarial example
    if wk == []:    
        print ("(5) no adversarial example is found in this round.")
        return (False,0,0,0,0)
    else: 
        print ("(5) an adversarial example has been found.")
        image0 = wk[0]
        eudist = euclideanDistance(image,image0)
        l1dist = l1Distance(image,image0)
        l0dist = l0Distance(image,image0)
        percent = diffPercent(image,image0)
        return (True,percent,eudist,l1dist,l0dist)
    
