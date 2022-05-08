import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path = r"Z:\CS4186\Assignment 1\gallery_4186"
querypath = r"Z:\CS4186\Assignment 1\cropped images"
images = []
classNames = []
filenames = []
queryList = os.listdir(querypath)

for query in queryList:
    myList = os.listdir(path)
    #print(myList)
    print("Total classes: "+ str(len(myList)))
    sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    #print(query)
    qpath = querypath + "\\"+ query
    queryimg = cv2.imread(qpath,cv2.IMREAD_GRAYSCALE)
    kpquery, desquery = sift.detectAndCompute(queryimg, None)
    kporb, desorb = orb.detectAndCompute(queryimg, None)

    dic = {}
    count = 0
    for cl in myList:
        #print(f'{path}\{cl}')
        #print(count)
        imgCur = cv2.imread(f'{path}\{cl}', cv2.IMREAD_GRAYSCALE)
        tempkp, tempdesc = sift.detectAndCompute(imgCur,None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desquery, tempdesc, k=2)
        good = 0
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good +=1

        tempkp1, tempdesc1 = orb.detectAndCompute(imgCur, None)
        matches2 = bf.knnMatch(desorb, tempdesc1, k=2)

        for m, n in matches2:
            if m.distance <0.75*n.distance:
                good+=1
        

        dic[cl] = good
        count+=1

    sorteddic = dict(sorted(dic.items(), key=lambda item: item[1],reverse=True))

    #fname = querypath.split("\\")[-1]
    fn = query.split(".")[0]
    name = fn + ".txt"
    outfile = open(name, "w+")
    outfile.write(fn+": ")

    i = 0
    for k, val in sorteddic.items():
        nam = k.split(".")[0]
        i = i+1
        #if(i<15):
            #print(nam)
        outfile.write(nam + " ")

    i = 0
    fnam = fn+"top10.txt"
    ofl = open(fnam, "w+")
    for k, val in sorteddic.items():
        #print(k, ":", val)            
        ofl.write(k+ " : "+ str(val)+ "\n")
        i=i+1
        if(i>10):
            break



    




