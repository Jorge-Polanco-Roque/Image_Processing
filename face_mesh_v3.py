import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import cv2 as cv

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20,50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []

blinkCounter = 0
counter = 0
color = (0,255,0)

#LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
#RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv.circle(img, face[id], 5, color, cv.FILLED)

        leftUp = face[159]
        leftdown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftdown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        cv.line(img, leftUp, leftdown, (0,200,0), 3)
        cv.line(img, leftLeft, leftRight, (0,200,0), 3)

        ratio = int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)

        if len(ratioList)>3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 40 and counter == 0:
            blinkCounter += 1
            color = (0,255,0)
            counter = 1
        
        if counter !=0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0,0,255)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50,100),
                            colorR=color)
                            
        imgPlot = plotY.update(ratio, color)

        img = cv.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    else:
        img = cv.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

        #for id in RIGHT_EYE:
        #    cv.circle(img, face[id], 5, (0,0,255), cv.FILLED)

        #for id in LEFT_EYE:
        #    cv.circle(img, face[id], 5, (0,0,255), cv.FILLED)
    
    
    cv.imshow("Image", imgStack)
    cv.waitKey(25)

