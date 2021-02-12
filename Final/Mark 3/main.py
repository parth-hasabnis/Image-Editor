#PyQT 5
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
import numpy
from matplotlib import pyplot as plt
import pywt
import pywt.data

#######################
import sys
import os
from PyQt5.uic import loadUiType
from os import path
from datetime import datetime
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw , ImageFont
from numpy.core.fromnumeric import resize


## Load UI FIle
FORM_CLASS,_=loadUiType(path.join(path.dirname(__file__),"editor.ui"))


class AppWindow(QMainWindow, FORM_CLASS):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.InitUi()
        self.handleMenu()
        self.handlestackedWidget()
        self.handleResizeWidget()
        self.handleStackedWidgetSeg()
        self.regApply.clicked.connect(self.registerImage)
        saveFileName = None

    def InitUi(self):
        self.HideAll()
        self.stackedWidgetSeg.hide()
        self.stackedWidgetSeg.setCurrentIndex(0)
        self.transformBox.hide()

    def handleResizeWidget(self):
        self.perSlider.setMinimum(1)
        self.perSlider.setMaximum(100)
        self.perSlider.setTickInterval(10)
        self.perSlider.setTickPosition(QSlider.TicksBelow)

        self.spinBoxWidth.setMinimum(100)
        self.spinBoxWidth.setMaximum(1920)
        self.spinBoxHeight.setMinimum(100)
        self.spinBoxHeight.setMaximum(1080)

        self.applyPerSize.clicked.connect(self.resizePercentage)

        self.perSlider.valueChanged.connect(lambda: self.perValue.setText(str(self.perSlider.value())))

    def handlestackedWidget(self):

        # button connect ----------------------------------------------
        # self.selectFilter.clicked.connect(self.chooseFilter)
        self.sobelApply.clicked.connect(self.filterSobel)
        self.averageApply.clicked.connect(self.filterAverage)
        self.gaussianApply.clicked.connect(self.filterGaussian)
        self.laplacianApply.clicked.connect(self.filterLaplacian)
        self.negativeApply.clicked.connect(self.filterNegative)
        self.sharpenApply.clicked.connect(self.filterSharpen)
        self.embossApply.clicked.connect(self.filterEmboss)
        self.sepiaApply.clicked.connect(self.filterSepia)

        self.sobelSlider.valueChanged.connect(lambda: self.sobelKernelValue.setText(str(2*self.sobelSlider.value()+1)))
        self.averageKernelSlider.valueChanged.connect(lambda: self.averageKernelSize.setText(str(2*self.averageKernelSlider.value()+1)))
        self.gaussianSlider.valueChanged.connect(lambda: self.gaussianKernelLabel.setText(str(2*self.gaussianSlider.value()+1)))
        self.dial.valueChanged.connect(lambda: self.dialLabel.setText(str(self.dial.value()/10)))


        self.filterBox.activated.connect(self.chooseFilter)

        # Widget init -------------------------------------------------
        self.sobelSlider.setMinimum(1)
        self.sobelSlider.setMaximum(10)
        self.sobelSlider.setTickInterval(1)
        self.sobelSlider.setTickPosition(QSlider.TicksBelow)

        self.averageKernelSlider.setMinimum(1)
        self.averageKernelSlider.setMaximum(10)
        self.averageKernelSlider.setTickInterval(2)
        self.averageKernelSlider.setTickPosition(QSlider.TicksBelow)

        self.gaussianSlider.setMinimum(1)
        self.gaussianSlider.setMaximum(10)
        self.gaussianSlider.setTickInterval(2)
        self.gaussianSlider.setTickPosition(QSlider.TicksBelow)

        self.dial.setMinimum(0)
        self.dial.setMaximum(100)
        self.dial.setNotchesVisible(True)

    def handleStackedWidgetSeg(self):
        self.fourierApply.clicked.connect(self.transformFourier)
        self.gaborApply.clicked.connect(self.transformGabor)
        self.RGBApply.clicked.connect(self.detectColorRGB)
        self.HSVApply.clicked.connect(self.detectColorHSV)
        self.waveletApply.clicked.connect(self.transformWavelet)

        self.sliderGaborKernel.valueChanged.connect(lambda: self.gaborKernelLabel.setText(str(2*self.sliderGaborKernel.value()+1)))
        self.sliderSigma.valueChanged.connect(lambda: self.gaborSigmaLabel.setText(str(self.sliderSigma.value())))
        self.gaborTheta.valueChanged.connect(lambda: self.gaborThetaLabel.setText(str(self.gaborTheta.value())))
        self.sliderR.valueChanged.connect(lambda: self.labelR.setText(str(self.sliderR.value())))
        self.sliderG.valueChanged.connect(lambda: self.labelG.setText(str(self.sliderG.value())))
        self.sliderB.valueChanged.connect(lambda: self.labelB.setText(str(self.sliderB.value())))

        self.transformBox.activated.connect(self.chooseTransform)

        # Widget init ------------------------------------------------------------------------------------------
        self.gaborTheta.setMinimum(0)
        self.gaborTheta.setMaximum(180)
        self.gaborTheta.setNotchesVisible(True)

        self.sliderSigma.setMinimum(1)
        self.sliderSigma.setMaximum(50)
        self.sliderSigma.setTickInterval(5)
        self.sliderSigma.setTickPosition(QSlider.TicksBelow)       

        self.sliderGaborKernel.setMinimum(1)
        self.sliderGaborKernel.setMaximum(10)
        self.sliderGaborKernel.setTickInterval(1)
        self.sliderGaborKernel.setTickPosition(QSlider.TicksBelow)   

        self.sliderR.setMinimum(0)
        self.sliderR.setMaximum(255)
        self.sliderG.setMinimum(0)
        self.sliderG.setMaximum(255)
        self.sliderB.setMinimum(0)
        self.sliderB.setMaximum(255)     

        self.lowH.setRange(0, 179)
        self.lowS.setRange(0, 255)
        self.lowV.setRange(0, 255)
        self.highH.setRange(0, 179)
        self.highS.setRange(0, 255)
        self.highV.setRange(0, 255)


    def handleMenu(self):
        self.actionNew.triggered.connect(self.LoadImages)
        self.actionFilter.triggered.connect(self.FilterImage)
        self.actionSave.triggered.connect(self.SaveImages)
        self.actionResize.triggered.connect(self.ResizeImage)
        self.actionApply_Transform.triggered.connect(self.transformImage)
        self.actionDetect_Color.triggered.connect(self.detectColor)
        self.actionLoad_Align_Image.triggered.connect(self.LoadAlign)
        self.actionSave_Image.triggered.connect(self.SaveAlign)

    def LoadImages(self):
        image, _ = QFileDialog.getOpenFileName(self, 'Upload Your Image ', r"C:\\Users\\Parth Hasabnis\\Pictures","Image files (*.jpg *.gif *.png)")
        global img
        global resizeImg
        global refImage
        img = cv2.imread(image, 1)
        resizeImg = img.copy()
        refImage = img.copy()
        self.imageLabel.setPixmap(QPixmap(image))
        self.imageLabelSegment.setPixmap(QPixmap(image))
        self.regRef.setPixmap(QPixmap(image))
        # self.transformBox.show()
        # self.stackedWidgetSeg.show()

    def LoadAlign(self):
        image, _ = QFileDialog.getOpenFileName(self, 'Upload Your Image ', r"C:\\Users\\Parth Hasabnis\\Pictures","Image files (*.jpg *.gif *.png)")
        global alignImage
        alignImage = cv2.imread(image, 1)
        self.regAlign.setPixmap(QPixmap(image))

    def SaveAlign(self):
        path, type = QFileDialog.getSaveFileName(self, "Save Your Image", r"C:\\Users\\Parth Hasabnis\\Pictures","Image files (*.jpg *.gif *.png)")
        cv2.imwrite(path, transformed_img)

    def SaveImages(self):
        path, type = QFileDialog.getSaveFileName(self, "Save Your Image", r"C:\\Users\\Parth Hasabnis\\Pictures","Image files (*.jpg *.gif *.png)")
        cv2.imwrite(path, self.saveFileName)

    def HideAll(self):
        self.stackedWidget.hide()
        self.filterBox.hide()
        self.resizeWidget.hide()
        self.transformBox.hide()
        self.stackedWidgetSeg.hide()

    def chooseFilter(self):
        if(self.filterBox.currentText() == "Gaussian Blur"):
            self.stackedWidget.setCurrentIndex(0)
        elif(self.filterBox.currentText() == "Sobel Edge"):
            self.stackedWidget.setCurrentIndex(1)
        elif(self.filterBox.currentText() == "Average"):
            self.stackedWidget.setCurrentIndex(2)
        elif(self.filterBox.currentText() == "Sharpen"):
            self.stackedWidget.setCurrentIndex(3)
        elif(self.filterBox.currentText() == "Laplacian"):
            self.stackedWidget.setCurrentIndex(4)
        elif(self.filterBox.currentText() == "Negative"):
            self.stackedWidget.setCurrentIndex(5)
        elif(self.filterBox.currentText() == "Emboss"):
            self.stackedWidget.setCurrentIndex(6)
        elif(self.filterBox.currentText() == "Sepia"):
            self.stackedWidget.setCurrentIndex(7)

    def chooseTransform(self):
        if(self.transformBox.currentText() == "Fourier"):
            self.stackedWidgetSeg.setCurrentIndex(0)
        elif(self.transformBox.currentText() == "Gabor"):
            self.stackedWidgetSeg.setCurrentIndex(1)
        elif(self.transformBox.currentText() == "Wavelet"):
            self.stackedWidgetSeg.setCurrentIndex(2)
        else:
            self.stackedWidgetSeg.setCurrentIndex(0)

    def filterSobel(self):  
        global edgeImage 
        edgeImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=(2*int(self.sobelSlider.value())+1)) # sobel x
        y_sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=(2*int(self.sobelSlider.value())+1)) # sobel y


        if(self.checkX.isChecked()==1 and self.checkY.isChecked()==0):
            edgeImage = x_sobel
        elif(self.checkY.isChecked()==1 and self.checkX.isChecked()==0):
            edgeImage = y_sobel
        elif(self.checkX.isChecked()==1 and self.checkY.isChecked()==1):
            G = list()
            for row in range(img.shape[0]):
                row_new = [(x_sobel[row, col]**2 + y_sobel[row,col]**2)**0.5 for col in range(img.shape[1])]
                G.append(row_new)
            G = numpy.array(G)
        
            edgeImage = G
        elif(self.checkX.isChecked()==0 and self.checkY.isChecked()==0):
            edgeImage = img

        resizeImg = edgeImage.copy()
        # im_pil = Image.fromarray(edgeImage)
        # im_pil = Image.fromarray(edgeImage.astype('uint8'), 'RGB')
        cv2.imwrite("Sobel.jpg", edgeImage)
        self.saveFileName = edgeImage
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Sobel.jpg"))
        # color_coverted = cv2.cvtColor(edgeImage, cv2.COLOR_BGR2RGB)
        # pil_image=Image.fromarray(color_coverted)
        # self.imageLabel.setPixmap(QPixmap(pil_image))

    def filterAverage(self):
        global averageImage
        size = 2*int(self.averageKernelSlider.value())+1
        averageImage = cv2.blur(img, (size, size))
        cv2.imwrite("Average.jpg", averageImage)
        self.saveFileName = averageImage
        resizeImg = averageImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Average.jpg"))
    
    def filterGaussian(self):
        global gaussianImage
        size = 2*int(self.gaussianSlider.value())+1
        sigma = self.dial.value()/10
        gaussianImage = cv2.GaussianBlur(src=img, ksize=(size, size), sigmaX=sigma, sigmaY=sigma)
        cv2.imwrite("Gaussian.jpg", gaussianImage)
        self.saveFileName = gaussianImage
        resizeImg = gaussianImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Gaussian.jpg"))

    def filterLaplacian(self):
        global laplaceImage
        laplaceImage = cv2.Laplacian(img, -1)
        cv2.imwrite("Laplacian.jpg", laplaceImage)
        self.saveFileName = laplaceImage
        resizeImg = laplaceImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Laplacian.jpg"))

    def filterSharpen(self):
        global sharpenImage
        filter = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpenImage=cv2.filter2D(img,-1,filter)
        cv2.imwrite("Sharpen.jpg", sharpenImage)
        self.saveFileName = sharpenImage
        resizeImg = sharpenImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Sharpen.jpg"))

    def filterNegative(self):
        global negativeImage
        negativeImage = cv2.bitwise_not(img)
        cv2.imwrite("Negative.jpg", negativeImage)
        self.saveFileName = negativeImage
        resizeImg = negativeImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Negative.jpg"))

    def filterEmboss(self):
        global embossImage
        filter = numpy.array([[0,1,0],[0,0,0],[0,-1,0]])
        embossImage = cv2.filter2D(img, -1, filter)
        embossImage = embossImage + 128
        cv2.imwrite("Emboss.jpg", embossImage)
        self.saveFileName = embossImage
        resizeImg = embossImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Emboss.jpg"))

    def filterSepia(self):
        global sepiaImage
        filter = numpy.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])
        sepiaImage = cv2.transform(img, filter)
        cv2.imwrite("Sepia.jpg", sepiaImage)
        self.saveFileName = sepiaImage
        resizeImg = sepiaImage.copy()
        self.imageLabel.setPixmap(QPixmap("D:/Image Editor/Final/Mark 3/Sepia.jpg"))

    def transformFourier(self):
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fourier = numpy.fft.fft2(img_)
        fshift = numpy.fft.fftshift(fourier)
        magnitude_spectrum = 20*numpy.log(numpy.abs(fshift))
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Fourier trasform ')
        plt.show()

    def transformGabor(self):
        theta = int(self.gaborTheta.value())
        theta = 3.142/180*theta
        sigma = int(self.sliderSigma.value())
        k = int(self.sliderGaborKernel.value())
        f = int(self.spinBoxFreq.value())

        kernel = cv2.getGaborKernel((k, k), sigma, theta, f, 1)
        gabor = cv2.filter2D(img, -1, kernel)
        plt.imshow(gabor)
        plt.title('Gabor Filtered image')
        plt.show()

    def transformWavelet(self):
        titles = [' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
        coeffs2 = pywt.dwt2(img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LH, HL, HH]):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()


    def FilterImage(self):
        self.HideAll()
        self.filterBox.show()
        self.stackedWidget.show()
        self.stackedWidget.setCurrentIndex(0)
        
        # self.stackedWidget.setCurrentWidget(self.home)

    def transformImage(self):
        self.transformBox.show()
        self.stackedWidgetSeg.show()
        self.stackedWidgetSeg.setCurrentIndex(0)

    def detectColor(self):
        self.HideAll()
        self.stackedWidgetSeg.show()
        self.stackedWidgetSeg.setCurrentIndex(4)

    def detectColorHSV(self):
        lowH = self.lowH.value()
        lowS = self.lowS.value()
        lowV = self.lowV.value()

        highH = self.highH.value()
        highS = self.highS.value()
        highV = self.highV.value()

        frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_HSV, (lowH, lowS, lowV), (highH, highS, highV))
        frame_threshold = cv2.bitwise_and(frame_HSV, frame_HSV, mask = mask)
        frame_threshold = cv2.cvtColor(frame_threshold, cv2.COLOR_HSV2RGB)
        plt.imshow(frame_threshold)
        #plt.title('Segmented Image')
        plt.show()

    def detectColorRGB(self):
        r = self.sliderR.value()
        g = self.sliderG.value()
        b = self.sliderB.value()

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = numpy.uint8([[[b, g, r]]])
        color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        '''
        lower = tuple(numpy.float32(color[0][0]))
        upper = tuple(numpy.float32(color[0][0]+1))
        print(color, lower, upper)
        print(type(lower))
        '''
        lower = (int(color[0][0][0]), int(color[0][0][1]), int(color[0][0][2]))
        upper = (int(color[0][0][0]+1), int(color[0][0][1]+1), int(color[0][0][2]+1))
        frame_threshold = cv2.inRange(frame, lower, upper)

        plt.imshow(frame_threshold, cmap='gray')
        #plt.title('Segmented Image')
        plt.show()

    def ResizeImage(self):
        self.HideAll()
        self.resizeWidget.show()
        self.spinBoxWidth.setValue(resizeImg.shape[1])
        self.spinBoxHeight.setValue(resizeImg.shape[0])

    def resizePercentage(self):
        scale = int(self.perSlider.value())
        width = int(resizeImg.shape[1] * scale / 100)
        height = int(resizeImg.shape[0] * scale / 100)
        dim = (width, height)
        resizedImg = cv2.resize(resizeImg, dim, interpolation = cv2.INTER_AREA)
        self.saveFileName = resizedImg

    def resizeDimension(self):
        width = int(self.spinBoxWidth.value())
        height = int(self.spinBoxHeight.value())
        dim = (width, height)
        resizedImg = cv2.resize(resizeImg, dim, interpolation = cv2.INTER_AREA)
        self.saveFileName = resizedImg

    def registerImage(self):
        img1 = cv2.cvtColor(alignImage, cv2.COLOR_BGR2GRAY) 
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        height, width = img2.shape 
        
        orb_detector = cv2.ORB_create(5000) 
         
        kp1, d1 = orb_detector.detectAndCompute(img1, None) 
        kp2, d2 = orb_detector.detectAndCompute(img2, None) 

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
        matches = matcher.match(d1, d2) 
 
        matches.sort(key = lambda x: x.distance) 

        matches = matches[:int(len(matches)*90)] 
        no_of_matches = len(matches) 
  
        p1 = numpy.zeros((no_of_matches, 2)) 
        p2 = numpy.zeros((no_of_matches, 2)) 
        
        for i in range(len(matches)): 
            p1[i, :] = kp1[matches[i].queryIdx].pt 
            p2[i, :] = kp2[matches[i].trainIdx].pt 
        
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

        global transformed_img
        transformed_img = cv2.warpPerspective(alignImage, 
                            homography, (width, height)) 
        transformed_img_2 = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(transformed_img_2)
        plt.title('Aligned Image')
        plt.show()

def main():
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()



