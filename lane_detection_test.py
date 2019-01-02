import numpy as np 
import cv2
from matplotlib import pyplot as plt 

# Creating subplot sub-routine
def create_subplot(image_list,figure_number=1):
    a = image_list[0]
    b = image_list[1]
    c = image_list[2]
    d = image_list[3]
    plt.figure(figure_number)
    plt.subplot(221),plt.imshow(a),plt.title('a')
    plt.subplot(222),plt.imshow(b),plt.title('b')
    plt.subplot(223),plt.imshow(c),plt.title('c')
    plt.subplot(224),plt.imshow(d),plt.title('d')
    plt.show()


# Importing the images
a = cv2.imread('test_images/Figure1a.jpg')
b = cv2.imread('test_images/Figure1b.jpg')
c = cv2.imread('test_images/Figure1c.jpg')
d = cv2.imread('test_images/Figure1d.jpg')
images = [a,b,c,d]
create_subplot(images)

# Free memory
a = 0
b = 0
c = 0 
d = 0

grey_images = []
bw_images = []
dilate_images = []
thin_images = []
lane_images = []
p1 = []
p2 = []

for image in images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_images.append(image)
create_subplot(grey_images)

for grey_image in grey_images:
    _,bw_image = cv2.threshold(grey_image,200,255,cv2.THRESH_BINARY)
    bw_images.append(bw_image)
create_subplot(bw_images)

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
for bw_image in bw_images:
    dilate_image = cv2.dilate(bw_image,kernel1,iterations = 1)
    dilate_images.append(dilate_image)
create_subplot(dilate_images)

# Free memory
bw_images = []
grey_images = []
grey_image = 0
bw_image = 0

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))    
for dilate_image in dilate_images:
    thin_image = cv2.erode(dilate_image,kernel2,iterations = 2)
    thin_images.append(thin_image)
create_subplot(thin_images)


#Hough transforms
i = 0
for thin_image in thin_images:
    lines = cv2.HoughLinesP(thin_image,
                            rho = 5,
                            theta = np.pi/90,
                            threshold = 40,
                            lines=np.array([]),
                            minLineLength = 10,
                            maxLineGap = 1000)
    for j in range(0,2):
        for x1,y1,x2,y2 in lines[j]:
            cv2.line(images[i],(x1,y1),(x2,y2),(0,255,0),5)
            p1.append((x1,y1))
            p2.append((x2,y2))
    pp1 = (int(p1[0][0] - (p1[0][0] - p2[1][0])/2), p1[0][1])
    pp2 = (int(p2[0][0] - (p2[0][0] - p1[1][0])/2), p2[0][1])
    cv2.line(images[i],pp1,pp2,(255,0,0),5)
    lane_images.append(images[i])
    i += 1
    p1 = []
    p2 =[]    
create_subplot(lane_images)