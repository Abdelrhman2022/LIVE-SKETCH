'''About the project : Today we are going to make a Real-time/ live Sketch 
making script using OpenCV in Python. OpenCV makes it very easy for us to work with 
images and videos on the computer. We will also make use of Numpy and Matplotlib to make this live sketch app.'''

#importing the opencv library 
import cv2
#importing the numpy library for working with image arrays
import numpy as np

# image to cartoon widgets

def edge_mask(img,ksize,block_size):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_median = cv2.medianBlur(gray,ksize)
    edges = cv2.adaptiveThreshold(gray_median,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,block_size,ksize)
    return edges

def kmeans_cluster(img,k):
    # transform image
    data = np.float32(img).reshape((-1,3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    # k means
    ret, label, center = cv2.kmeans(data,k,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def adjust_gamma(image,gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i/255)**invGamma)*255 for i in np.arange(0,256)]) # lookup table
    lut_img = cv2.LUT(image.astype(np.uint8),table.astype(np.uint8))
    return lut_img

def cartoon_image(img):
    ksize=5
    block_size=7
    k=7
    d=7
    sigmaspace=200
    sigmacolor=200
    # step-1: edge_mask, kmeans
    edgeMask =  edge_mask(img,ksize,block_size)
    cluster_img = kmeans_cluster(img,k)
    # step-2: apply bilateral filter
    bilateral = cv2.bilateralFilter(cluster_img,d=d,sigmaColor=sigmacolor,sigmaSpace=sigmaspace)
    cartoon = cv2.bitwise_and(bilateral,bilateral,mask=edgeMask)
    
    return cartoon
    
def pencil(img):
    #pencil sketch
    ksize = 21
    sigmaX = 9
    gamma = 0.1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray,(ksize,ksize),sigmaX) # ksize = 3 to 25 and sigmax = 1 to 15
    gray_blur_divide = cv2.divide(gray,gray_blur,scale=256)
    pencil_sktech = adjust_gamma(gray_blur_divide,gamma=gamma)#0 - 1
    
    return pencil_sktech

def sketch(image):
    #scale = 0.40
    #height of the image
    height=int(image.shape[0])
    
    #width of image
    width=int(image.shape[1])
    
    #storing the image dimension
    dim=(width,height)
    
    #resize the image into our own dimension
    resize=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    
    #applying a kernel
    '''Kernels in computer vision are matrices, used to perform some kind of convolution in our data.
    Let’s try to break this down.
    Convolutions are mathematical operations between two functions that create a third function. 
    In image processing, it happens by going through each pixel to perform a calculation with the pixel and its neighbours.
    The kernels will define the size of the convolution, the weights applied to it, and an anchor point usually positioned at the center.'''
    kernel=np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    
    #sharpning the resized image
    '''Applying the sharpening filter will sharpen the edges in the image. 
    This filter is very useful when we want to enhance the edges in an image that's not crisp.'''
    sharp=cv2.filter2D(resize,-1,kernel)
    
    #converting the image into gray scale
    gray=cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
    inv=255-gray
    
    
    #apply bluring
    '''In Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. 
    The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced.'''
    blur=cv2.GaussianBlur(src=inv,ksize=(15,15),sigmaX=0,sigmaY=0)
    #draw sketch
    
    s=cv2.divide(gray,255-blur,scale=256)
    return s

'''cap = cv.VideoCapture(0)

VideoCapture() Function is used to capture video either from the camera or already recorded video. cap variable returns a boolean value (True if able to retrieve/capture video successfully, False if not able to successfully capture the video). It takes one parameter:

0 – Front Camera
1 – Rear Camera
If the Cap returns True, then read() function is applied to it and it returns two things:

Boolean Value (Was it successfully able to read the frame, If yes)
Returns the frame of the video.
Each Frame is sent to a sketch() function that takes frame as input parameter and manipulates it to return sketch of the frame.

Don’t forget to release the captured video at the end of the while loop. Otherwise, it will consume all your machine’s memory.'''

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    
    cv2.imshow('Live_Sketch',pencil(frame))
    cv2.imshow('Live_image',frame)
    #cv2.imshow('Live_Cartoon',cartoon_image(frame))
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
