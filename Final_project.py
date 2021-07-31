from tkinter import *
from tkinter.font import Font
from tkinter.ttk import Combobox
from PIL import ImageTk,Image

from tkinter import filedialog
import numpy as np
import cv2

# DO NOT DELETE ANYTHING 

#http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter
#https://realpython.com/python-gui-tkinter/

from time import sleep
def delete():
    my_image_label.destroy()
    btn.destroy()
    
def load2d():
    global my_image_label
    global my_image
    global btn
    global fname
    global orig_img
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(("bmp files", "*.bmp"),("all files", "*.*"),("png files", "*.png"), ("jpeg files", "*.jpeg")))
    fname = root.filename
    orig_img = cv2.imread(fname)
    imS = cv2.resize(orig_img, (512, 512)) #can change size here
    cv2.imshow("INPUT Image",imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Vinutha J
#code to Apply histogram equalization - CONTRAST ENHANCEMENT 
def ApplyHistEqua():
    global fname
    img = cv2.imread(fname)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    imS = cv2.resize(img_output, (512, 512)) #can change size here
    cv2.imshow("Histogram Equalized Image",imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#code to Apply sepia filter
def ApplySepia():
    global fname
    img = cv2.imread(fname)
    #original = img.copy()
    imS = np.array(img, dtype=np.float64) # converting to float to prevent loss
    imS = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    imS[np.where(imS > 255)] = 255 # normalizing values greater than 255 to 255
    imS = np.array(imS, dtype=np.uint8)# converting back to int
    imS = cv2.resize(imS, (512, 512))
    cv2.imshow("Sepia", imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Add functions, buttons, text accordingly for remaining filters 
def ApplyGrayScale():
    global fname
    img = cv2.imread(fname)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_img = cv2.resize(grey_img, (512,512))
    cv2.imshow("GrayScale",grey_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def ApplyPencilSketch():
    global fname
    img = cv2.imread(fname)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(gray_img)
    smoothing_img = cv2.GaussianBlur(invert_img, (21, 21),sigmaX=0, sigmaY=0)
    final_img = dodgeV2(gray_img,smoothing_img)
    final_img = cv2.resize(final_img, (512,512))
    cv2.imshow("Pencil Sketch",final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    
def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

#Shreyas KG
def ApplyBlur():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    imgBlur = cv2.blur(img,(3,3))
    cv2.imshow("Blurred Image",imgBlur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ApplyColouredEdges():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    imgBlur = cv2.blur(img,(5,5))
    #canny filter for edge detection
    imgCanny = cv2.Canny(imgBlur,50,50)
    imgColouredEdge = cv2.bitwise_and(imgBlur,imgBlur,mask = imgCanny)
    cv2.imshow("Coloured Edges",imgColouredEdge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def ImagePlanes():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    blue,green,red = cv2.split(img)
    cv2.imshow("Red",red)
    cv2.imshow("Green",green)
    cv2.imshow("Blue",blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Abhishek P Ramesh
def vignette():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    height, width = img.shape[:2]
    print(height,width)
    # Create a Gaussian filter

    kernel_x = cv2.getGaussianKernel(width,200)
    kernel_y = cv2.getGaussianKernel(height,200)
    kernel = kernel_y * kernel_x.T


    filter = 255 * kernel / np.linalg.norm(kernel)
    print(filter)
    vig_img= np.copy(img)

    for i in range(3):

        vig_img[:,:,i] = vig_img[:,:,i] * filter

    cv2.imshow("vignette",vig_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# negative image
def negative():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))

    img_neg = 1-img
    cv2.imshow("negative",img_neg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Log transformation
def log():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    
    # s = c* log(r+1)
    # c = L / (log10(1+L))
    c = 255 / (np.log10(1+255))

    s = c*(np.log10(img+1))

    s = np.array(s,dtype='uint8')
    cv2.imshow("Log Transformed",s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Sainath Biradar
def Contours():
    global fname;
    img = cv2.imread(fname)
    img = cv2.resize(img,(512,512))
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(f"Number of contours : {len(contours)}")
    print(contours[0])
          
    cv2.drawContours(img,contours,-1,(0,255,0),3)
    cv2.drawContours(imgray,contours,-1,(0,255,0),3)
    
    cv2.imshow("Image",img)
    cv2.imshow("Image GRAY",imgray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
root = Tk()
root.configure(bg = 'palegreen')
root.title('FCV Coding assignment')
root.resizable(width = TRUE, height = TRUE)
root.geometry("1000x1000")
font1 = Font(family = "Times", size = 20, weight = "bold", underline = 1)
font2 = Font(family = "Times", size = 13, weight = "bold")
frame1 = Frame(root, width = 480, height = 30, pady = 5, bg = "palegreen").grid(row = 0, columnspan = 2)
label1 = Label(frame1, text = "Shodhaka-Tantramsha:", font = font1, fg = "black", bg = "lightblue1", width = 50, height = 2, relief = 'solid', bd = 1 )
label1.grid(row =0,padx=100) #Shodhaka-Tantramsha: (samskrutham) => filtering software

label2 = Label( frame1, text = " ", fg = "black", bg = "palegreen")
label2.grid(row = 1,column =0)

label3 = Label(frame1, text = "Load input Image: ", font = font2, fg = "black", bg = "wheat1", anchor=SE, height = 2, width = 25)
label3.grid(row = 2,column =0)

label4 = Label( frame1, text = "Apply Histogram Equalization: ", font = font2, fg = "black", bg = "khaki1", anchor=SE, height = 2, width = 25)
label4.grid(row = 3,column =0)

label5 = Label( frame1, text = "Apply sepia ", font = font2, fg = "black", bg = "lavender", anchor=SE, height = 2, width = 25)
label5.grid(row = 4) # providing labels 

label6 = Label(frame1, text = "Apply Gray Scale ", font = font2, fg = "black", bg = "#FEFE58", anchor = SE, height =2, width = 25)
label6.grid(row = 5)

label7 = Label(frame1, text = "Apply Pencil Sketch ", font = font2, fg = "black", bg = "#9AFE58", anchor = SE, height =2, width = 25)
label7.grid(row = 5)

label7 = Label(frame1, text = "Blur(3X3) ", font = font2, fg = "black", bg = "#FF7F7F", anchor = SE, height =2, width = 25)
label7.grid(row = 6)

label7 = Label(frame1, text = "Coloured Edges ", font = font2, fg = "black", bg = "#32CD32", anchor = SE, height =2, width = 25)
label7.grid(row = 7)

label7 = Label(frame1, text = "R,G,B Planes ", font = font2, fg = "black", bg = "#ADD8E6", anchor = SE, height =2, width = 25)
label7.grid(row = 8)

label7 = Label(frame1, text = "Vignette ", font = font2, fg = "black", bg = "cyan", anchor = SE, height =2, width = 25)
label7.grid(row = 9)

label7 = Label(frame1, text = "Log ", font = font2, fg = "black", bg = "yellow", anchor = SE, height =2, width = 25)
label7.grid(row = 10)

label7 = Label(frame1, text = "Negative ", font = font2, fg = "black", bg = "magenta", anchor = SE, height =2, width = 25)
label7.grid(row = 11)
          
label7 = Label(frame1, text = "Contours ", font = font2, fg = "black", bg = "gray", anchor = SE, height =2, width = 25)
label7.grid(row = 12)

#call functions here + button

btn_load_3d = Button(frame1, text="Open", height = 1, width = 20, command=load2d).grid(row = 2, column = 1)
    
btn_load_3d = Button(frame1, text="Apply HE", height = 1, width = 20, command=ApplyHistEqua).grid(row = 3, column = 1)

btn_load_3d = Button(frame1, text="See sepia ", height = 1, width = 20, command=ApplySepia).grid(row = 4, column = 1)

btn_load_3d = Button(frame1, text="Gray Scale ", height = 1, width = 20, command=ApplyGrayScale).grid(row = 5, column = 1)

btn_load_3d = Button(frame1, text="Pencil Sketch ", height = 1, width = 20, command=ApplyPencilSketch).grid(row = 5, column = 1)

btn_load_3d = Button(frame1, text="Blur ", height = 1, width = 20, command=ApplyBlur).grid(row = 6, column = 1)

btn_load_3d = Button(frame1, text="Colored Edge ", height = 1, width = 20, command=ApplyColouredEdges).grid(row = 7, column = 1)

btn_load_3d = Button(frame1, text="Image planes ", height = 1, width = 20, command = ImagePlanes).grid(row = 8, column = 1)

btn_load_3d = Button(frame1, text="Vignette", height = 1, width = 20, command = vignette).grid(row = 9, column = 1)

btn_load_3d = Button(frame1, text="Log ", height = 1, width = 20, command = log).grid(row = 10, column = 1)

btn_load_3d = Button(frame1, text="Image Negative ", height = 1, width = 20, command = negative).grid(row = 11, column = 1)
          
btn_load_3d = Button(frame1, text="Detect Contours ", height = 1, width = 20, command = Contours).grid(row = 12, column = 1)
root.mainloop()
