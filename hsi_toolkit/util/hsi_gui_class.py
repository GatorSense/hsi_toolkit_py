'''
Use this GUI if you want to compare 2 hyperspectral images with binary masks at the same time.
This GUI compares to 2 HSI images at the same time
You will need to edit the code from line 76 to line 176.
Give it a minute once you finished selecting the files
Click on the 'Help!' button and read the guide to understand the features
'''

import tkinter as tk
from tkinter import messagebox, Scrollbar
from PIL import Image, ImageTk, ImageDraw
from tkinter import ttk
from tkinter import filedialog
import numpy as np
from spectral import imshow, view_cube
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spectral.io.envi as envi
import cv2
import pandas as pd

# Create a Tkinter window; Zoomed means full screen
root = tk.Tk()
root.state('zoomed')
root.title('Pixel Information')
root.pack_propagate(False) # root will not resize itself

# Create a frame to contain the canvas and scrollbar
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a canvas widget
canvas = tk.Canvas(main_frame, bg="white")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a vertical scrollbar linked to the canvas
vsb = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
vsb.pack(side=tk.RIGHT, fill=tk.Y)

# Configure canvas to work with scrollbar
canvas.configure(yscrollcommand=vsb.set)

canvasFrame = tk.Frame(canvas, bg = 'white')
canvas.create_window((0, 0), window=canvasFrame, anchor="nw")

# Divide the canvas into two parts
canvasFrame.columnconfigure(0, weight=1)
canvasFrame.columnconfigure(1, weight=3)

# Sets the frames dimensions based on window screen
frame1_width = (root.winfo_screenwidth() * 2) // 5
frame1_height = root.winfo_screenheight()
frame2_width = (root.winfo_screenwidth() * 3) // 5
frame2_height = root.winfo_screenheight()


# Create a frame and a canvas for the two parts
frame1 = tk.Canvas(canvasFrame, bg="white", width=frame1_width, height=frame1_height)
frame2 = tk.Frame(canvasFrame, bg="lightgrey", width=frame2_width, height=frame2_height)

#This divides frame1 into 2 parts for both hsi images

canvas1 = tk.Canvas(frame1, bg="white", width=frame1_width, height=frame1_height/2 - 10)
canvas1.grid(row = 0, column= 0, sticky = 'nsew')
half = tk.Canvas(frame1, bg="lightgrey", width=frame1_width, height=frame1_height/2 - 10)
half.grid(row = 1, column= 0, sticky = 'nsew')

# Place the frames in the divided parts
frame1.grid(row=0, column=0, sticky="nsew") #nsew means frame expands to all sides
frame2.grid(row=0, column=1, sticky="nsew")

canvasFrame.update_idletasks()  # Ensure the canvas dimensions are updated
canvas.config(scrollregion=canvas.bbox("all"))

'''
USER NEEDS TO CHANGE: FILE PATH LOCATION, HOW THEY UPLOAD THEIR DATA, AND RGB VALUES BASED ON THEIR DATA 
'''
# This is for the first HSI Image

# Gets the SWIR data
# This is the aligned SWIR data which is stored as a .npy file. 
SWIR_Path = filedialog.askopenfilename(initialdir= '/anika/HSI image', title = "SWIR Data", filetypes = (('npy files', '*.npy'),('all files', '*.*'),))
SWIR_Data = np.load(SWIR_Path) [:,5:] 

# Gets the VNIR Mask
mask_path = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "VNIR Mask", filetypes = (('npy files', '*.npy'),('all files', '*.*'),))
VNIR_Mask = np.load(mask_path)
rf,cf = np.where(VNIR_Mask==1) #used to crop the image to focus on region of interest

# Gets the VNIR Data
data_hdr = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data.hdr", filetypes = (('hdr files', '*.hdr'),('all files', '*.*'),))
data = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data", filetypes = (('all files', '*.*'),))

# You can change the way the data is being uploaded if necesarry. 
hsi_ref = envi.open(data_hdr,data)
hsi_np = np.copy(hsi_ref.asarray())[:,:,:-44] #TAKE AWAY LAST 44 BANDS

# Constructs data cube with VNIR and SWIR data
zero_Matrix = np.zeros((hsi_np.shape[0],hsi_np.shape[1],SWIR_Data.shape[1]))
zero_Matrix[rf,cf] = SWIR_Data
wholeCube = np.dstack((hsi_np,zero_Matrix))
updatedCube = (wholeCube[rf,cf])
wholeCube_cropped = wholeCube[min(rf):max(rf),min(cf):max(cf)]

# Gets data to construct RGB Image
red_band = hsi_np[:, :, 167]  # Replace with the correct band index
green_band = hsi_np[:, :,87]  # Replace with the correct band index
blue_band = hsi_np[:, :, 50]  # Replace with the correct band index

# Normalize the bands to [0, 1] or [0, 255] (adjust based on your data type)
red_band_normalized = (red_band - red_band.min()) / (red_band.max() - red_band.min())
green_band_normalized = (green_band - green_band.min()) / (green_band.max() - green_band.min())
blue_band_normalized = (blue_band - blue_band.min()) / (blue_band.max() - blue_band.min())

# Create an RGB image
image = np.power(np.dstack((red_band_normalized, green_band_normalized, blue_band_normalized))*255,1)
realIm = (np.array(image)[min(rf):max(rf),min(cf):max(cf)]).astype(np.uint8)
RGB_image = Image.fromarray(realIm)
ImageArr = np.array(wholeCube_cropped)
tk_image = ImageTk.PhotoImage(RGB_image)



# This is for the second HSI Image

# Gets the SWIR data
# This is the aligned SWIR data which is stored as a .npy file. 
SWIR_Path2 = filedialog.askopenfilename(initialdir= '/anika/HSI image', title = "SWIR Data", filetypes = (('npy files', '*.npy'),('all files', '*.*'),))
SWIR_Data2 = np.load(SWIR_Path2) [:,5:] 

# Gets the VNIR Mask
mask_path2 = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "VNIR Mask", filetypes = (('npy files', '*.npy'),('all files', '*.*'),))
VNIR_Mask2 = np.load(mask_path2)
rf2,cf2 = np.where(VNIR_Mask2==1) #used to crop the image to focus on region of interest

# Gets the VNIR Data
data_hdr2 = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data.hdr", filetypes = (('hdr files', '*.hdr'),('all files', '*.*'),))
data2 = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data", filetypes = (('all files', '*.*'),))
# You can change the way the data is being uploaded if necesarry. 
hsi_ref2 = envi.open(data_hdr2,data2)
hsi_np2 = np.copy(hsi_ref2.asarray())[:,:,:-44] #TAKE AWAY LAST 44 BANDS

# Constructs data cube with VNIR and SWIR data
zero_Matrix2 = np.zeros((hsi_np2.shape[0],hsi_np2.shape[1],SWIR_Data2.shape[1]))
zero_Matrix2[rf2,cf2] = SWIR_Data2
wholeCube2 = np.dstack((hsi_np2,zero_Matrix2))
updatedCube2 = (wholeCube2[rf2,cf2])
wholeCube_cropped2 = wholeCube2[min(rf2):max(rf2),min(cf2):max(cf2)]

# Gets data to construct RGB Image
red_band2 = hsi_np2[:, :, 167]  # Replace with the correct band index
green_band2 = hsi_np2[:, :,87]  # Replace with the correct band index
blue_band2 = hsi_np2[:, :, 50]  # Replace with the correct band index

# Normalize the bands to [0, 1] or [0, 255] (adjust based on your data type)
red_band_normalized2 = (red_band2 - red_band2.min()) / (red_band2.max() - red_band2.min())
green_band_normalized2 = (green_band2 - green_band2.min()) / (green_band2.max() - green_band2.min())
blue_band_normalized2 = (blue_band2 - blue_band2.min()) / (blue_band2.max() - blue_band2.min())

# Create an RGB image
image2 = np.power(np.dstack((red_band_normalized2, green_band_normalized2, blue_band_normalized2))*255,1)
realIm2 = (np.array(image2)[min(rf2):max(rf2),min(cf2):max(cf2)]).astype(np.uint8)
RGB_image2 = Image.fromarray(realIm2)
ImageArr2 = np.array(wholeCube_cropped2)
tk_image2 = ImageTk.PhotoImage(RGB_image2)

# User can change line color and alpha value for line plots
global linecolor
linecolor = 'green'
global linecolor2
linecolor2 = 'orange'
global alphaV
alphaV = 0.5

'''
USER SHOULD NOT NEED TO CHANGE ANY CODE AFTER THIS POINT
'''

# Creates canvas3 to create a scollable area for image and creates canvas 2 to hold the image
canvas3 = tk.Canvas(canvas1, width = frame1_width, height = 730/2)
canvas3.grid(row = 0, column= 0, sticky = 'nsew')
canvas2 = tk.Canvas(canvas3, width = RGB_image.width, height = RGB_image.height)
canvas2.grid(row = 0, column= 0, sticky = 'nsew')
canvas2.create_image(0,0, anchor = tk.NW, image = tk_image)
canvas2.config(scrollregion=canvas2.bbox(tk.ALL))

# Configure scrollbars for canvas3 (this is to view entire image if image is to big to fit on screen)
vscrollbar2 = ttk.Scrollbar(canvas1, orient=tk.VERTICAL, command=canvas3.yview)
vscrollbar2.grid(row=0, column=3, sticky="ns")
canvas3.configure(yscrollcommand=vscrollbar2.set)
hscrollbar2 = ttk.Scrollbar(canvas1, orient=tk.HORIZONTAL, command=canvas3.xview)
hscrollbar2.grid(row=1, column=0, sticky="ew")
canvas3.configure(xscrollcommand=hscrollbar2.set)

# Configure canvas3 to scroll with canvas2 inside it
canvas3.create_window((0, 0), window=canvas2, anchor="nw")

# Configure scroll region for canvas2 and canvas3
canvas2.update_idletasks()  # Update canvas2 to get correct bbox
canvas3.config(scrollregion=canvas2.bbox(tk.ALL))
canvas1.config(scrollregion=(0, 0,frame1_width, 730/2))




#Create a second canvas for the second HSI image
canvas4 = tk.Canvas(half, width = frame1_width, height = 730/2)
canvas4.grid(row = 1, column= 0, sticky = 'nsew')
canvas5 = tk.Canvas(canvas4, width = RGB_image2.width, height = RGB_image2.height)
canvas5.grid(row = 1, column= 0, sticky = 'nsew')
canvas5.create_image(0,0, anchor = tk.NW, image = tk_image2)
canvas5.config(scrollregion=canvas5.bbox(tk.ALL))

# Configure scrollbars for canvas4 (this is to view entire image if image is to big to fit on screen)
vscrollbar3 = ttk.Scrollbar(half, orient=tk.VERTICAL, command=canvas4.yview)
vscrollbar3.grid(row=1, column=3, sticky="ns")
canvas4.configure(yscrollcommand=vscrollbar3.set)
hscrollbar3 = ttk.Scrollbar(half, orient=tk.HORIZONTAL, command=canvas4.xview)
hscrollbar3.grid(row=2, column=0, sticky="ew")
canvas4.configure(xscrollcommand=hscrollbar3.set)

# Configure canvas4 to scroll with canvas5 inside it
canvas4.create_window((0, 0), window=canvas5, anchor="nw")

# Configure scroll region for canvas4 and canvas5
canvas5.update_idletasks()  # Update canvas2 to get correct bbox
canvas4.config(scrollregion=canvas5.bbox(tk.ALL))
half.config(scrollregion=(0, 0,frame1_width, 730))

# Create a Matplotlib figure and display a plot in the second frame
fig, axs = plt.subplots(4, 1, figsize=(frame2_width/100  , frame2_height/70 - 10))
index = np.concatenate((np.linspace(400,1000,372)[:-44],np.linspace(900,2500,270)[5:]))
axs[0].set_title('Random Pixels')
axs[1].set_title('Average Pixel')
axs[2].set_title('Selected Pixels')
axs[3].set_title('All Pixels')
for i in range(0,4):
    axs[i].set_ylim(0,1.2)
    axs[i].set_yticks(np.linspace(0,1,5))
    axs[i].set_xlabel('Spectral Band (nm)')
    axs[i].set_ylabel('Reflectance')
plt.subplots_adjust(hspace= 0.8) # adds spacing between the subplots


# to place the plots on tkinter canvas
canvas_matplotlib = FigureCanvasTkAgg(fig, master=frame2)
canvas_matplotlib.draw()
canvas_matplotlib.get_tk_widget().pack(fill = tk.BOTH, expand=True)



'''
THIS SECTION OF THE CODE IS FOR THE FIRST HSI IMAGE
'''


#Plots the mean RGB Values
mean2 = np.mean(updatedCube,axis= 0)
global mean1
mean1, = axs[1].plot(index, mean2, color = linecolor,alpha=alphaV)

'''
THIS IS ALL THE CODE FOR THE RANDOM PIXEL GRAPH (FIRST PLOT)
'''
randomNum = 6
randLineTracker = {}
randData = []
coordinates = []

# THIS FUNCTION WILL PLOT RANDOM PIXELS AND STORE INFROMATION IN LISTS/DICTS DEFINED BELOW
def randomPixels(ImageShape):
    global randomNum # number of pixels to be randomly plotted
    global randList
    global randData
    randList = [] # stores pixel coordinates
    count = 1
    for pixel in range (0,int(randomNum)):
        pixelNum = np.random.randint(0,ImageShape[0])
        data = updatedCube[pixelNum]
        randData.append(data)
        randList.append([cf[pixelNum], rf[pixelNum]])
        #gives you a line 2D object with properties of the line
        line, = axs[0].plot(index, data, color = linecolor,alpha=alphaV)
        axs[0].annotate(str(count), (2.05,data[-1]), color = linecolor) # adds the numbers to the end of the lines plotted
        randLineTracker[count] = line
        count += 1
        fig.canvas.draw()

# Calls the function intially
randomPixels(updatedCube.shape)

# CLEARS THE RANDOM PIXEL PLOT AND GRAPHS NEW PIXELS 
def updatePlotforRand():
     global randLineTracker
     for lines in randLineTracker.values():
         lines.remove()
     randLineTracker = {}
     axs[0].set_title('Random Pixels')
     axs[0].set_ylim(0,1.2)
     axs[0].set_yticks(np.linspace(0,1,5))
     axs[0].set_xlabel('Spectral Band (nm)')
     axs[0].set_ylabel('Reflectance')
     global randData
     randData = []
     fig.canvas.draw() 
     randomPixels(updatedCube.shape)  

# These 2 dicts hold the information on dots and numbers plotted on image
randDots = {} 
textInfo2 = {}

# GRAPHS THE DOTS AND NUMBERS OF THE RANDOM PIXELS ON THE IMAGE
def showRandPixels():
    global textInfo2
    global randDots 
    #Hides the selected pixel information
    canvas2.itemconfigure("selectedDots", state = "hidden")
    canvas2.itemconfigure("text", state = "hidden")

    for coordinates in all_Dots:
        canvas2.delete(all_Dots[coordinates])
    for textID, (textX, textY) in all_text.items():
        canvas2.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if randDots != {} and textInfo2 != {}:
        for coordinates in randDots:
            canvas2.delete(randDots[coordinates])
        for textID, (textX, textY) in textInfo2.items():
            canvas2.delete(textID)
    #This will plot the new dots and numbers on the image
    Count = 1
    for pixel in randList:
        normalized_cf = int(pixel[0] - min(cf))
        normalized_rf = int(pixel[1] - min(rf))
        x = normalized_cf
        y = normalized_rf
        rand_dot = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = linecolor, tag = "random_selectedDots")
        randDots[(x,y)] = rand_dot
        rand_textID = canvas2.create_text(x+15, y, text = str(Count), fill = linecolor, tag = "random_text")
        textInfo2[rand_textID] = (x+15,y)
        Count += 1

# FINDS WHICH PIXEL CORRESPONDS TO THE LINE THE USER CLICKED ON
def whichRandomPixel():
    global randDots #contains the pixel coordinate and the dots drawn
    global yCoords2 #contains HSI data values of the selected pixels

    for coordinates in all_Dots:
        canvas2.delete(all_Dots[coordinates])
    for textID, (textX, textY) in all_text.items():
        canvas2.delete(textID)
    for keys in randDots.keys():
        x,y = keys
        pixelkey = y,x
        rgbPixel = ImageArr[y][x]
        #trying to find the pixel clicked on and enlarge the dot and number
        if np.array_equal(rgbPixel,yCoords2):
            global selectedDot2
            selectedDot2 = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText.insert(tk.END, "Pixel Coordinates = " + str(pixelkey) )

#CHECKS TO SEE IF YOU CLICKED ON A LINE IN THE RANDOM PIXEL PLOT
def clickRand_Selected(event):
    global randData
    if event.inaxes == axs[0]: #checks to see if you clicked in the first plot
        for key, lineOb in randLineTracker.items():
            contains, _ = lineOb.contains(event) #checks to see if an event happened near a line
            #contains is a true or false value based on if event happened on a line
            if contains:
                global yCoords2 # has rdg values of that line
                yCoords2 = randData[key-1]
                whichRandomPixel() #function will draw a dot and print the pixel coordinate and rgb values
                break

#DELETES THE WHITE DOT AND NUMBER
def NotRand_Selected(event):
    global selectedDot2
    canvas2.delete(selectedDot2)
    PixelText.delete('1.0', tk.END)

#Gets the user input for number of random pixels to be graphed
def get_input():
    global randomNum
    randomNum = entry.get()
    updatePlotforRand()

'''
THIS IS ALL THE CODE FOR THE SELECTED PIXEL GRAPH (THRID PLOT)
'''
ycoordList = [] #list stores all rgb values that are plotted
lineTracker = {} #stores line properties

#GRAPHS PIXEL THE USER CLICKS ON 
def get_pixel_rgb(event):
    global dotCount
    x, y = event.x, event.y
    data = ImageArr[y,x]
    ycoordList.append(data)
    line, = axs[2].plot(index,data, color = linecolor,alpha=alphaV)
    lineTracker[dotCount] = line
    axs[2].annotate(str(dotCount), (2.05,data[-1]), color = linecolor)
    fig.canvas.draw()

#THIS FUNCTION CLEARS THE SELECTED PIXEL PLOTS
def updateSelected4Dots(): 
     global lineTracker
     for lines in lineTracker.values():
         lines.remove()
     lineTracker = {}
     global ycoordList
     ycoordList = []
     axs[2].set_title('Selected Pixels')
     axs[2].set_ylim(0,1.2)
     axs[2].set_yticks(np.linspace(0,1,5))
     axs[2].set_xlabel('Spectral Band (nm)')
     axs[2].set_ylabel('Reflectance')
     fig.canvas.draw()

#THIS WILL UPDATE THE GRAPH IF THE USER UNSELECTS A PIXEL
def redraw_pixel_rgb(list_to_redraw):
    updateSelected4Dots()
    global dotCount
    dotCount = 1
    for coordinate in list_to_redraw:
        x,y = coordinate
        data = ImageArr[y,x]
        ycoordList.append(data)
        line, = axs[2].plot(index,data, color = linecolor,alpha=alphaV)
        lineTracker[dotCount] = line
        axs[2].annotate(str(dotCount), (2.05,data[-1]), color = linecolor)
        dotCount += 1
        fig.canvas.draw()


imagedots = {} #stores coordinates and dot properties
textInfo={} #stores text properties and coordinates
dotCount = 1 #keeps track of how mmany dots are plotted

#CHECKS TO SEE IF PIXEL ALREADY HAS A DOT AND NUMBER PLOTTED
#IF NOT IT GRAPHS THE DOT AND NUMBER. IF IT DOES, IT WILL DELETE THE DOT AND NUMBER
def dots(event):
    global dotCount
    x,y = event.x, event.y
    coordinates = (x,y)
    if coordinates in imagedots:
        canvas2.delete(imagedots[coordinates])
        for textID, (textX, textY) in textInfo.items():
            if textX == x+10 and textY == y:
                canvas2.delete(textID)
                del textInfo[textID]
                break
        for keys in lineTracker.keys():
            if keys == imagedots[coordinates]:
                del lineTracker[dotCount]
                break
        dotCount += -1
        del imagedots[coordinates]
        listRedraw = list(imagedots.keys())
        redraw_pixel_rgb(listRedraw) #redraws graph if user unselects a pixel
    else:
        dot = canvas2.create_oval(x-3,y-3,x+2, y+2, fill = linecolor, tag = "selectedDots")
        imagedots[coordinates] = dot
        textID = canvas2.create_text(x+10, y, text = str(dotCount), fill = linecolor, tag = "text")
        textInfo[textID] = (x+10,y)
        dotCount += 1

#ENLARGES THE DOT THAT CORRESPONDS TO THE LINE THE USER CLICKED ON
global selectedDot
def whichPixel():
    for keys in imagedots.keys():
        x,y = keys
        pixelkey = y,x
        dataPixel = ImageArr[y][x]
        if np.array_equal(dataPixel,yCoords):
            global selectedDot
            for key,item in textInfo.items():
                if item == (x+10,y):
                    text_value = canvas2.itemcget(key,'text')
                    global showText
                    showText = canvas2.create_text(x+10, y, text = str(text_value), fill = 'white')
            selectedDot = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText.insert(tk.END, "Pixel Coordinates = " + str(pixelkey)) #prints the picel coordinates in the textbox

#FINDS THE RGB VALUES OF THE LINE THE USER CLICKED ON
def clickSelected(event):
    global clicked
    clicked = 1
    if event.inaxes == axs[2]:
        for key, lineOb in lineTracker.items():
            contains, _ = lineOb.contains(event)
            if contains:
                global yCoords
                yCoords = ycoordList[key-1]
                whichPixel()
                break

#REMOVES THE SELECTED DOT AND TEXT AND REMOVES THE TEXT
def NotSelected(event):
    global selectedDot
    global showText
    canvas2.delete(selectedDot)
    canvas2.delete(showText)
    PixelText.delete('1.0', tk.END)
   
#THIS CLEARS THE SELECTED DOTS GRAPH, CLEARS THE IMAGE, AND EMPTIES ALL LISTS/DICTS
def updatePlot():
     global lineTracker
     for lines in lineTracker.values():
        lines.remove()
     lineTracker = {}
     allLines = axs[2].get_lines()
     for line in allLines:
         if line.get_color() == linecolor:
             line.remove()
     axs[2].set_title('Selected Pixels')
     axs[2].set_ylim(0,1.2)
     axs[2].set_yticks(np.linspace(0,1,5))
     axs[2].set_xlabel('Spectral Band (nm)')
     axs[2].set_ylabel('Reflectance')
     #clears the image by redrawing it
     global RGB_image,tk_image
     RGB_image = Image.fromarray(realIm) 
     tk_image = ImageTk.PhotoImage(RGB_image)
     canvas2.create_image(0, 0, anchor=tk.NW, image=tk_image) 
     #Clears all of the lists and dicts that keep track of selected pixels
     global ycoordList
     global coordinates
     coordinates = []
     ycoordList = []
     fig.canvas.draw()
     global dotCount
     dotCount = 1

#THIS SHOWS THE SELECTED DOTS ON THE IMAGE AND DELETES THE RANDOM DOTS
def showSelectedPixels():
    global textInfo2
    canvas2.itemconfigure("selectedDots", state = "normal")
    canvas2.itemconfigure("text", state = "normal")
    for coordinates in randDots:
        canvas2.delete(randDots[coordinates])
    for textID, (textX, textY) in textInfo2.items():
        canvas2.delete(textID)
    for coordinates in all_Dots:
        canvas2.delete(all_Dots[coordinates])
    for textID, (textX, textY) in all_text.items():
        canvas2.delete(textID)
    textInfo2 = {}


#PLOTS THE DOTS AND NUMBER BASED ON WHAT LINE THE USER CLICKS ON
def combinedFunctions(event):
    get_pixel_rgb(event)
    dots(event)

#Functions for click and Drag
last_x,  last_y = None, None
pen_color = "red"  # Change this to your desired color
pen_size = 1 #default pen size

#gets the pen size value that the user selects
def penValue(value):
    global pen_size
    pen_size = int(value)

 #Allows the user to draw on image based on a right click and drag motion
def on_button_press(event):
    global last_x,last_y
    last_x, last_y = event.x, event.y

def on_button_motion( event):
    global last_x,last_y
    if last_x is not None and last_y is not None:
            # Draw on the image
            draw_on_image(event.x, event.y)
            last_x, last_y = event.x, event.y
            global coordinates
            coordinates.append((event.x,event.y))

#Plots the spectra for pixels that were drew over
def graphDraggedPixel(list_to_redraw):
    # updateSelected4Dots()
    global dotCount
    dotCount = 1
    for coordinate in list_to_redraw:
        x,y = coordinate
        data = ImageArr[y,x]
        ycoordList.append(data)
        line, = axs[2].plot(index,data, color = linecolor,alpha=alphaV)
        lineTracker[dotCount] = line
        axs[2].annotate(str(dotCount), (2.05,data[-1]), color = linecolor)
        dotCount += 1
        fig.canvas.draw()

# Creates dots on the pixels that the user drew over
def on_button_release(event):
    global last_x,last_y
    global dotCount
    last_x, last_y = None, None
    global coordinates
    graphDraggedPixel(coordinates)
    for x,y in coordinates:
        if (x,y) not in imagedots:
            coord = (x,y)
            dot = canvas2.create_oval(x-3,y-3,x+2, y+2, fill = linecolor, tag = "selectedDots")
            imagedots[coord] = dot
            textID = canvas2.create_text(x+10, y, text = str(dotCount), fill = linecolor, tag = "text")
            textInfo[textID] = (x+10,y)
            dotCount += 1

# Allows the user to draw on the image    
def draw_on_image( x, y):
        global tk_image, RGB_image
        global last_x,last_y, pen_size
        # Draw on the image using PIL
        draw = ImageDraw.Draw(RGB_image)
        draw.line([last_x, last_y, x, y], fill=pen_color, width=pen_size)
        # Update the canvas
        tk_image = ImageTk.PhotoImage(RGB_image)
        canvas2.create_image(0, 0, anchor=tk.NW, image=tk_image) 

# Allows user to interact with image
canvas2.bind("<Button-1>", combinedFunctions) #left click to select a simple pixel

# Right click and drag to draw on image
canvas2.bind("<ButtonPress-3>", on_button_press)
canvas2.bind("<B3-Motion>", on_button_motion)
canvas2.bind("<ButtonRelease-3>", on_button_release)

'''
THIS SECTION IS FOR THE ALL PIXELS PLOT
'''
# THIS FUNCTION GRAPHS EVERY <BLANK> PIXEL ON THE 4th PLOT
def everyPixel():
    #empties all lists and dicts that keep track of pixel info
    global stepsize, PixelList, allData, allLinetracker
    allData = []
    PixelList= []
    all_count = 1
    #resets the 4th graph

    allLinetracker = {}
    allLines = axs[3].get_lines()
    for line in allLines:
        if line.get_color() == linecolor:
            line.remove()
    axs[3].set_title('All Pixels')
    axs[3].set_ylim(0,1.2)
    axs[3].set_yticks(np.linspace(0,1,5))
    axs[3].set_xlabel('Spectral Band (nm)')
    axs[3].set_ylabel('Reflectance')
    fig.canvas.draw()

    #graphs every <blank> pixels
    for pixelR in range(0, updatedCube.shape[0], int(stepsize)):
        PixelList.append([cf[pixelR], rf[pixelR]])
        data = updatedCube[pixelR]
        allData.append(data)
        all_line, = axs[3].plot(index, data, color = linecolor,alpha=alphaV)
        allLinetracker[all_count] = all_line
        axs[3].annotate(str(all_count), (2.05,data[-1]), color = linecolor) # adds the numbers to the end of the lines plotted
        all_count += 1
    fig.canvas.draw()

# These 2 dicts hold the information on dots and numbers plotted on image for every Xth pixel
all_Dots = {} 
all_text = {}

def showAllPixels():
    global all_text
    global all_Dots 
    #Hides the selected pixel information
    canvas2.itemconfigure("selectedDots", state = "hidden")
    canvas2.itemconfigure("text", state = "hidden")
    canvas2.itemconfigure('random_selectedDots', state = 'hidden')
    canvas2.itemconfigure('random_text', state = 'hidden')
    if randDots != {} and textInfo2!= {}:
        for coordinates in randDots:
            canvas2.delete(randDots[coordinates])
        for textID, (textX, textY) in textInfo2.items():
            canvas2.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if all_Dots != {} and all_text != {}:
        for coordinates in all_Dots:
            canvas2.delete(all_Dots[coordinates])
        for textID, (textX, textY) in all_text.items():
            canvas2.delete(textID)
    #This will plot the new dots and numbers on the image
    Count = 1
    for pixel in PixelList:
        x = pixel[0] - min(cf)
        y = pixel[1] - min(rf)
        all_dot = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = linecolor, tag = "all_selectedDots")
        all_Dots[(x,y)] = all_dot
        all_textID = canvas2.create_text(x+15, y, text = str(Count), fill = linecolor, tag = "all_text")
        all_text[all_textID] = (x+15,y)
        Count += 1

    canvas_matplotlib.mpl_connect('button_press_event', clickAll_Selected)
    canvas_matplotlib.mpl_connect('button_release_event', NotAll_Selected)

# FINDS WHICH PIXEL CORRESPONDS TO THE LINE THE USER CLICKED ON
def whichAllPixel():
    global all_Dots #contains the pixel coordinate and the dots drawn
    global selectedPixel #contains rgb values of the selected pixels
    global showTextAll
    for keys in all_Dots.keys():
        x,y = keys
        pixelkey = y,x
        dataPixel = ImageArr[y][x]
        #trying to find the pixel clicked on and enlarge the dot and number
        if np.array_equal(dataPixel,selectedPixel):
            global selectedDotAll
            selectedDotAll = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText.insert(tk.END, "Pixel Coordinates = " + str(pixelkey) )


#CHECKS TO SEE IF YOU CLICKED ON A LINE IN THE ALL PIXEL PLOT
def clickAll_Selected(event):
    global clickedAll
    global allLinetracker
    global allData
    if event.inaxes == axs[3]: #checks to see if you clicked in the first plot
        for key, lineOb in allLinetracker.items():
            contains, _ = lineOb.contains(event) #checks to see if an event happened near a line
            #contains is a true or false value based on if event happened on a line
            if contains:
                global selectedPixel # has rdg values of that line
                selectedPixel = allData[key-1]
                whichAllPixel() #function will draw a dot and print the pixel coordinate and rgb values
                break

#DELETES THE WHITE DOT AND NUMBER
def NotAll_Selected(event):
    global selectedDotAll
    global showTextAll
    canvas2.delete(selectedDotAll)
    # canvas2.delete(showTextAll)
    PixelText.delete('1.0', tk.END)

#Gets the user input for every <blank> pixel to be graphed
def inputForStep():
    global stepsize
    stepsize = entry2.get()
    everyPixel()

'''
THIS IS FOR SELECTING WHICH COMPONENT TO PLOT
'''
# GET USER INPUT FOR WHICH COMPONENT TO VISUALIZE
def get_componentNum():
    global component
    component = entry3.get()
    updatePlotforComp()
    graphRandComp()

# THIS IS TO GRAPH RANDOM PIXELS OF THAT COMPONENT
def graphRandComp():
    global component, randomNum, randList, randData
    randList = [] # stores pixel coordinates
    # clear dots of all pixels plot id its on the image
    for coordinates in all_Dots:
        canvas2.delete(all_Dots[coordinates])
    for textID, (textX, textY) in all_text.items():
        canvas2.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if randDots != {} and textInfo2 != {}:
        for coordinates in randDots:
            canvas2.delete(randDots[coordinates])
        for textID, (textX, textY) in textInfo2.items():
            canvas2.delete(textID)
    #gets pixels that correspond to the component number the user selects
    count = 1
    output = cv2.connectedComponentsWithStats(VNIR_Mask.astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    compR, compC = np.where(labels == int(component))
    global RGB_image, tk_image
    compCube = (wholeCube[compR,compC])
    #plots those pixels spectra on the 1st graph
    for pixel in range (0,int(randomNum)):
        pixelNum = np.random.randint(0,compCube.shape[0])
        data = compCube[pixelNum]
        randData.append(data)
        randList.append([compC[pixelNum], compR[pixelNum]])
        #gives you a line 2D object with properties of the line
        line, = axs[0].plot(index, data, color = linecolor,alpha=alphaV)
       
        axs[0].annotate(str(count), (2.05,data[-1]), color = linecolor) # adds the numbers to the end of the lines plotted
        randLineTracker[count] = line
        count += 1
        fig.canvas.draw()
    # Updates the labeling on the average pixels plot 
    global mean1
    mean1.remove()
    axs[1].set_title('Average of Component ' + str(component))
    mean = np.mean(compCube,axis= 0)
    mean1, = axs[1].plot(index, mean, color = linecolor,alpha=alphaV)
    axs[1].set_ylim(0,1.2)
    axs[1].set_yticks(np.linspace(0,1,5))
    axs[1].set_xlabel('Spectral Band (nm)')
    axs[1].set_ylabel('Reflectance')
    fig.canvas.draw()

# CLEARS THE RANDOM PIXEL PLOT AND GRAPHS NEW PIXELS 
def updatePlotforComp():
     global randLineTracker
     for lines in randLineTracker.values():
         lines.remove()
     randLineTracker = {}
     axs[0].set_title('Random Pixels')
     axs[0].set_ylim(0,1.2)
     axs[0].set_yticks(np.linspace(0,1,5))
     axs[0].set_xlabel('Spectral Band (nm)')
     axs[0].set_ylabel('Reflectance')
     global randData
     randData = []
     

#help messagebox that lists all of the feature information
def help():
    messagebox.showinfo('Guide' , "'Clear Selected Pixels' : Removes the dots that the user selected"+'\n' +  '\n' +"'Generate Random Pixels' : Plots new random pixel spectras"+'\n' +  '\n' +"'Show Random Pixels' : Only displays dots corresponding to the random pixel plot"
                        +'\n' +  '\n' + "'Show Selected Pixels' : Only displays dots for the Selected Pixels plot " +'\n' + '\n' + "'Show All Pixels': Only displays dots for every X Pixels"+'\n' + '\n' + "'Graph Every <blank> Pixels' : Plots every X number of pixels"
                        +'\n' + '\n' +  "'Enter Random Number of Pixels' : User can choose the number of pixels to be plotted in the Random Pixels plot"+'\n' + '\n' + "'Enter Component Number': Select which component you want to generate random pixels and graph average spectra" +'\n' + '\n'  + "'Export' : Create an excel sheet with the selected pixel coordinates and HSI data"
                        +'\n' + '\n' +  "'Slider' : Chooses the pen's thickness" + '\n' + 'To select a region of pixels, right click and drag.' +'\n' + '\n' + "'Clear Image' : Remove Drawings and dots from the image"
                        +'\n' + '\n' + 'Left click on the image to graph the corresponding pixel spectra. You can click on any spectra in the 1st, 3rd, and 4th plots to get the pixel coordinates'
                         +'\n' + '\n' + 'Use the scroll bars to view the image if the image is too big to fit the canvas')
    

# Function that saves pixel coordinates and data in Excel
def export():
    data = []
    for coordinates in imagedots.keys():
        info = (coordinates[1], coordinates[0], *(wholeCube[coordinates[1],coordinates[0]]))
        data.append(info)
    df = pd.DataFrame(data)
    columns = ['Row', 'Column', 'Data'] + [''] * (df.shape[1] - 3)
    df.columns = columns
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                             filetypes=[("Excel files", "*.xlsx"), 
                                                        ("All files", "*.*")])
    if file_path:
        # Save the DataFrame to an Excel file
        df.to_excel(file_path, index=False)

 #Clears image of all dots and pixel selections
def clear(): 
    global RGB_image,tk_image
    RGB_image = Image.fromarray(realIm) 
    tk_image = ImageTk.PhotoImage(RGB_image)
    canvas2.create_image(0, 0, anchor=tk.NW, image=tk_image) 

# THIS ALLOWS THE USER TO INTERACT WITH THE GRAPHS
canvas_matplotlib.mpl_connect('button_press_event', clickSelected)
canvas_matplotlib.mpl_connect('button_release_event', NotSelected)
canvas_matplotlib.mpl_connect('button_press_event', clickRand_Selected)
canvas_matplotlib.mpl_connect('button_release_event', NotRand_Selected)

# All of the Displays and their commands for the first hsi image
frame_grid = tk.Frame(canvas1, bg = 'white') #creates a grid to place all widgets
frame_grid.grid(row=2, column=0, columnspan=4, sticky="ew")

button = tk.Button(frame_grid, text="Clear Selected Pixels", command = updatePlot )
button.grid(row = 0, column=0)

button2 = tk.Button(frame_grid, text="Generate Random Pixels", command = updatePlotforRand)
button2.grid(row = 0, column=1)

button4 = tk.Button(frame_grid, text="Show Random Pixels",command = showRandPixels)
button4.grid(row = 0, column=2)

entry2 = tk.Entry(frame_grid, width = 20)
entry2.grid(row = 1, column=0)

button3 = tk.Button(frame_grid, text= "Graph Every <blank> Pixel", command = inputForStep)
button3.grid(row = 1, column=1)

entry = tk.Entry(frame_grid, width=20)
entry.grid(row = 1, column=2)

button5 = tk.Button(frame_grid, text="Enter Random Pixel Number", command=get_input)
button5.grid(row = 1, column=3)

clearbutton = tk.Button(frame_grid, text="Clear Image", command=clear)
clearbutton.grid(row = 1, column=4)

button6 = tk.Button(frame_grid, text="Show Selected Pixels", command = showSelectedPixels)
button6.grid(row = 0, column=3)

button7 = tk.Button(frame_grid, text="Show All Pixels", command = showAllPixels)
button7.grid(row = 0, column=4)

entry3 = tk.Entry(frame_grid, width = 20)
entry3.grid(row = 2, column=0)

button8 = tk.Button(frame_grid, text="Enter Component Number", command = get_componentNum)
button8.grid(row = 2, column=1)

export_button = tk.Button(frame_grid, text= "Export File", command = export) 
export_button.grid(row=2, column=2) 

slider = tk.Scale(frame_grid, from_= 1, to_= 8, orient=tk.HORIZONTAL, command = penValue)
slider.grid (row = 2, column=3)

help = tk.Button(frame_grid, text="Help!", command = help)
help.grid(row = 2, column=4)

PixelText = tk.Text(canvas1, height = 1, width = 40)
PixelText.grid(row = 3, column=0)






'''
REST OF THE CODE IS FOR THE SECOND HSI IMAGE
'''




#Plots the mean RGB Values
mean2 = np.mean(updatedCube2,axis= 0)
global mean_2
mean_2, = axs[1].plot(index, mean2, color = linecolor2,alpha=alphaV)

'''
THIS IS ALL THE CODE FOR THE RANDOM PIXEL GRAPH (FIRST PLOT)
'''
randomNum2 = 6
randLineTracker2 = {}
randData2= []
coordinates2 = []

# THIS FUNCTION WILL PLOT RANDOM PIXELS AND STORE INFROMATION IN LISTS/DICTS DEFINED BELOW
def randomPixels2(ImageShape):
    global randomNum2 # number of pixels to be randomly plotted
    global randList2
    global randData2
    randList2 = [] # stores pixel coordinates2
    count = 1
    for pixel in range (0,int(randomNum2)):
        pixelNum = np.random.randint(0,ImageShape[0])
        data = updatedCube2[pixelNum]
        randData2.append(data)
        randList2.append([cf2[pixelNum], rf2[pixelNum]])
        #gives you a line 2D object with properties of the line
        line, = axs[0].plot(index, data, color = linecolor2,alpha=alphaV)
        axs[0].annotate(str(count), (2.05,data[-1]), color = linecolor2) # adds the numbers to the end of the lines plotted
        randLineTracker2[count] = line
        count += 1
        fig.canvas.draw()

# Calls the function intially
randomPixels2(updatedCube2.shape)

# CLEARS THE RANDOM PIXEL PLOT AND GRAPHS NEW PIXELS 
def updatePlotforRand2():
     global randLineTracker2
     for lines in randLineTracker2.values():
         lines.remove()
     randLineTracker2 = {}
     axs[0].set_title('Random Pixels')
     axs[0].set_ylim(0,1.2)
     axs[0].set_yticks(np.linspace(0,1,5))
     axs[0].set_xlabel('Spectral Band (nm)')
     axs[0].set_ylabel('Reflectance')
     global randData2
     randData2= []
     fig.canvas.draw() 
     randomPixels2(updatedCube2.shape)  

# These 2 dicts hold the information on dots2 and numbers plotted on image
randDots2 = {} 
textInfo4 = {}

# GRAPHS THE dots2 AND NUMBERS OF THE RANDOM PIXELS ON THE IMAGE
def showRandPixels2():
    global textInfo4
    global randDots2 
    #Hides the selected pixel information
    canvas5.itemconfigure("selectedDots", state = "hidden")
    canvas5.itemconfigure("text", state = "hidden")

    for coordinates2 in all_Dots2:
        canvas5.delete(all_Dots2[coordinates2])
    for textID, (textX, textY) in all_text2.items():
        canvas5.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if randDots2 != {} and textInfo4 != {}:
        for coordinates2 in randDots2:
            canvas5.delete(randDots2[coordinates2])
        for textID, (textX, textY) in textInfo4.items():
            canvas5.delete(textID)
    #This will plot the new dots2 and numbers on the image
    Count = 1
    for pixel in randList2:
        normalized_cf = int(pixel[0] - min(cf2))
        normalized_rf = int(pixel[1] - min(rf2))
        x = normalized_cf
        y = normalized_rf
        rand_dot = canvas5.create_oval(x-5,y-5,x+5, y+5, fill = linecolor2, tag = "random_selectedDots")
        randDots2[(x,y)] = rand_dot
        rand_textID = canvas5.create_text(x+15, y, text = str(Count), fill = linecolor2, tag = "random_text")
        textInfo4[rand_textID] = (x+15,y)
        Count += 1

# FINDS WHICH PIXEL CORRESPONDS TO THE LINE THE USER clicked2 ON
def whichRandomPixel2():
    global randDots2 #contains the pixel coordinate and the dots2 drawn
    global yCoords2 #contains HSI data values of the selected pixels

    for coordinates2 in all_Dots2:
        canvas5.delete(all_Dots2[coordinates2])
    for textID, (textX, textY) in all_text2.items():
        canvas5.delete(textID)
    for keys in randDots2.keys():
        x,y = keys
        pixelkey = y,x
        rgbPixel = ImageArr2[y][x]
        #trying to find the pixel clicked2 on and enlarge the dot and number
        if np.array_equal(rgbPixel,yCoords2):
            global selectedDot2
            selectedDot2 = canvas5.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText_2.insert(tk.END, "Pixel coordinates2 = " + str(pixelkey) )

#CHECKS TO SEE IF YOU clicked2 ON A LINE IN THE RANDOM PIXEL PLOT
def clickRand_Selected2(event):
    global randData2
    if event.inaxes == axs[0]: #checks to see if you clicked2 in the first plot
        for key, lineOb in randLineTracker2.items():
            contains, _ = lineOb.contains(event) #checks to see if an event happened near a line
            #contains is a true or false value based on if event happened on a line
            if contains:
                global yCoords2 # has rdg values of that line
                yCoords2 = randData2[key-1]
                whichRandomPixel2() #function will draw a dot and print the pixel coordinate and rgb values
                break

#DELETES THE WHITE DOT AND NUMBER
def NotRand_Selected2(event):
    global selectedDot2
    canvas5.delete(selectedDot2)
    PixelText_2.delete('1.0', tk.END)

#Gets the user input for number of random pixels to be graphed
def get_input2():
    global randomNum2
    randomNum2 = entry_2.get()
    updatePlotforRand2()

'''
THIS IS ALL THE CODE FOR THE SELECTED PIXEL GRAPH (THRID PLOT)
'''
ycoordList2 = [] #list stores all rgb values that are plotted
lineTracker2 = {} #stores line properties

#GRAPHS PIXEL THE USER CLICKS ON 
def get_pixel_rgb2(event):
    global dotCount2
    x, y = event.x, event.y
    data = ImageArr2[y,x]
    ycoordList2.append(data)
    line, = axs[2].plot(index,data, color = linecolor2,alpha=alphaV)
    lineTracker2[dotCount2] = line
    axs[2].annotate(str(dotCount2), (2.05,data[-1]), color = linecolor2)
    fig.canvas.draw()

#THIS FUNCTION CLEARS THE SELECTED PIXEL PLOTS
def updateSelected4Dots2(): 
     
     global ycoordList2
     ycoordList2 = []
     global lineTracker2
     for lines in lineTracker2.values():
         lines.remove()
     lineTracker2 = {}
     axs[2].set_title('Selected Pixels')
     axs[2].set_ylim(0,1.2)
     axs[2].set_yticks(np.linspace(0,1,5))
     axs[2].set_xlabel('Spectral Band (nm)')
     axs[2].set_ylabel('Reflectance')
     fig.canvas.draw()

#THIS WILL UPDATE THE GRAPH IF THE USER UNSELECTS A PIXEL
def redraw_pixel_rgb2(list_to_redraw):
    updateSelected4Dots2()
    global dotCount2
    dotCount2 = 1
    for coordinate in list_to_redraw:
        x,y = coordinate
        data = ImageArr2[y,x]
        ycoordList2.append(data)
        line, = axs[2].plot(index,data, color = linecolor2,alpha=alphaV)
        lineTracker2[dotCount2] = line
        axs[2].annotate(str(dotCount2), (2.05,data[-1]), color = linecolor2)
        dotCount2 += 1
        fig.canvas.draw()


imagedots2 = {} #stores coordinates2 and dot properties
textInfo3={} #stores text properties and coordinates2
dotCount2 = 1 #keeps track of how mmany dots2 are plotted

#CHECKS TO SEE IF PIXEL ALREADY HAS A DOT AND NUMBER PLOTTED
#IF NOT IT GRAPHS THE DOT AND NUMBER. IF IT DOES, IT WILL DELETE THE DOT AND NUMBER
def dots2(event):
    global dotCount2
    x,y = event.x, event.y
    coordinates2 = (x,y)
    if coordinates2 in imagedots2:
        canvas5.delete(imagedots2[coordinates2])
        for textID, (textX, textY) in textInfo3.items():
            if textX == x+10 and textY == y:
                canvas5.delete(textID)
                del textInfo3[textID]
                break
        for keys in lineTracker2.keys():
            if keys == imagedots2[coordinates2]:
                del lineTracker2[dotCount2]
                break
        dotCount2 += -1
        del imagedots2[coordinates2]
        listRedraw = list(imagedots2.keys())
        redraw_pixel_rgb2(listRedraw) #redraws graph if user unselects a pixel
    else:
        dot = canvas5.create_oval(x-3,y-3,x+2, y+2, fill = linecolor2, tag = "selectedDots")
        imagedots2[coordinates2] = dot
        textID = canvas5.create_text(x+10, y, text = str(dotCount2), fill = linecolor2, tag = "text")
        textInfo3[textID] = (x+10,y)
        dotCount2 += 1

#ENLARGES THE DOT THAT CORRESPONDS TO THE LINE THE USER clicked2 ON
global selectedDot2
def whichPixel2():
    for keys in imagedots2.keys():
        x,y = keys
        pixelkey = y,x
        dataPixel = ImageArr2[y][x]
        if np.array_equal(dataPixel,yCoords2):
            global selectedDot2
            for key,item in textInfo3.items():
                if item == (x+10,y):
                    text_value = canvas5.itemcget(key,'text')
                    global showText2
                    showText2 = canvas5.create_text(x+10, y, text = str(text_value), fill = 'white')
            selectedDot2 = canvas5.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText_2.insert(tk.END, "Pixel coordinates2 = " + str(pixelkey)) #prints the picel coordinates2 in the textbox

#FINDS THE RGB VALUES OF THE LINE THE USER clicked2 ON
def clickSelected2(event):
    global clicked2
    clicked2 = 1
    if event.inaxes == axs[2]:
        for key, lineOb in lineTracker2.items():
            contains, _ = lineOb.contains(event)
            if contains:
                global yCoords2
                yCoords2 = ycoordList2[key-1]
                whichPixel2()
                break

#REMOVES THE SELECTED DOT AND TEXT AND REMOVES THE TEXT
def NotSelected2(event):
    global selectedDot2
    global showText2
    canvas5.delete(selectedDot2)
    canvas5.delete(showText2)
    PixelText_2.delete('1.0', tk.END)
   
#THIS CLEARS THE SELECTED dots2 GRAPH, CLEARS THE IMAGE, AND EMPTIES ALL LISTS/DICTS
def updatePlot2():
     global lineTracker2
     for lines in lineTracker2.values():
        lines.remove()
     lineTracker2 = {}
     allLines = axs[2].get_lines()
     for line in allLines:
         if line.get_color() == linecolor2:
             line.remove()
     axs[2].set_title('Selected Pixels')
     axs[2].set_ylim(0,1.2)
     axs[2].set_yticks(np.linspace(0,1,5))
     axs[2].set_xlabel('Spectral Band (nm)')
     axs[2].set_ylabel('Reflectance')
     #clears the image by redrawing it
     global RGB_image2,tk_image2
     RGB_image2 = Image.fromarray(realIm2) 
     tk_image2 = ImageTk.PhotoImage(RGB_image2)
     canvas5.create_image(0, 0, anchor=tk.NW, image=tk_image2) 
     #Clears all of the lists and dicts that keep track of selected pixels
     global ycoordList2
     global coordinates2
     coordinates2 = []
     ycoordList2 = []
     fig.canvas.draw()
     global dotCount2
     dotCount2 = 1

#THIS SHOWS THE SELECTED dots2 ON THE IMAGE AND DELETES THE RANDOM dots2
def showSelectedPixels2():
    global textInfo4
    canvas5.itemconfigure("selectedDots", state = "normal")
    canvas5.itemconfigure("text", state = "normal")
    for coordinates2 in randDots2:
        canvas5.delete(randDots2[coordinates2])
    for textID, (textX, textY) in textInfo4.items():
        canvas5.delete(textID)
    for coordinates2 in all_Dots2:
        canvas5.delete(all_Dots2[coordinates2])
    for textID, (textX, textY) in all_text2.items():
        canvas5.delete(textID)
    textInfo4 = {}


#PLOTS THE dots2 AND NUMBER BASED ON WHAT LINE THE USER CLICKS ON
def combinedFunctions2(event):
    get_pixel_rgb2(event)
    dots2(event)

#Functions for click and Drag
last_x2,  last_y2= None, None
pen_color2 = "red"  # Change this to your desired color
pen_size2 = 1 #default pen size

#gets the pen size value that the user selects
def penValue2(value):
    global pen_size2
    pen_size2 = int(value)

 #Allows the user to draw on image based on a right click and drag motion
def on_button_press2(event):
    global last_x2,last_y2
    last_x2, last_y2= event.x, event.y

def on_button_motion2( event):
    global last_x2,last_y2
    if last_x2 is not None and last_y2 is not None:
            # Draw on the image
            draw_on_image2(event.x, event.y)
            last_x2, last_y2= event.x, event.y
            global coordinates2
            coordinates2.append((event.x,event.y))

#Plots the spectra for pixels that were drew over
def graphDraggedPixel2(list_to_redraw):
    # updateSelected4Dots2()
    global dotCount2
    dotCount2 = 1
    for coordinate in list_to_redraw:
        x,y = coordinate
        data = ImageArr2[y,x]
        ycoordList2.append(data)
        line, = axs[2].plot(index,data, color = linecolor2,alpha=alphaV)
        lineTracker2[dotCount2] = line
        axs[2].annotate(str(dotCount2), (2.05,data[-1]), color = linecolor2)
        dotCount2 += 1
        fig.canvas.draw()

# Creates dots2 on the pixels that the user drew over
def on_button_release2(event):
    global last_x2,last_y2
    global dotCount2
    last_x2, last_y2= None, None
    global coordinates2
    graphDraggedPixel2(coordinates2)
    for x,y in coordinates2:
        if (x,y) not in imagedots2:
            coord = (x,y)
            dot = canvas5.create_oval(x-3,y-3,x+2, y+2, fill = linecolor2, tag = "selectedDots")
            imagedots2[coord] = dot
            textID = canvas5.create_text(x+10, y, text = str(dotCount2), fill = linecolor2, tag = "text")
            textInfo3[textID] = (x+10,y)
            dotCount2 += 1

# Allows the user to draw on the image    
def draw_on_image2( x, y):
        global tk_image2, RGB_image2
        global last_x2,last_y2, pen_size2
        # Draw on the image using PIL
        draw = ImageDraw.Draw(RGB_image2)
        draw.line([last_x2, last_y2, x, y], fill=pen_color2, width=pen_size2)
        # Update the canvas
        tk_image2 = ImageTk.PhotoImage(RGB_image2)
        canvas5.create_image(0, 0, anchor=tk.NW, image=tk_image2) 

# Allows user to interact with image
canvas5.bind("<Button-1>", combinedFunctions2) #left click to select a simple pixel

# Right click and drag to draw on image
canvas5.bind("<ButtonPress-3>", on_button_press2)
canvas5.bind("<B3-Motion>", on_button_motion2)
canvas5.bind("<ButtonRelease-3>", on_button_release2)

'''
THIS SECTION IS FOR THE ALL PIXELS PLOT
'''
# THIS FUNCTION GRAPHS EVERY <BLANK> PIXEL ON THE 4th PLOT
def everyPixel2():
    #empties all lists and dicts that keep track of pixel info
    global stepsize2, PixelList2, allData2, allLinetracker2
    allData2 = []
    
    allLinetracker2 = {}
    PixelList2= []
    all_count2 = 1
    #resets the 4th graph
    allLines = axs[3].get_lines()
    for line in allLines:
        if line.get_color() == linecolor2:
            line.remove()
    axs[3].set_title('All Pixels')
    axs[3].set_ylim(0,1.2)
    axs[3].set_yticks(np.linspace(0,1,5))
    axs[3].set_xlabel('Spectral Band (nm)')
    axs[3].set_ylabel('Reflectance')
    fig.canvas.draw()

    #graphs every <blank> pixels
    for pixelR in range(0, updatedCube2.shape[0], int(stepsize2)):
        PixelList2.append([cf2[pixelR], rf2[pixelR]])
        data = updatedCube2[pixelR]
        allData2.append(data)
        all_line, = axs[3].plot(index, data, color = linecolor2,alpha=alphaV)
        allLinetracker2[all_count2] = all_line
        axs[3].annotate(str(all_count2), (2.05,data[-1]), color = linecolor2) # adds the numbers to the end of the lines plotted
        all_count2 += 1
    fig.canvas.draw()

# These 2 dicts hold the information on dots2 and numbers plotted on image for every Xth pixel
all_Dots2 = {} 
all_text2 = {}

def showAllPixels2():
    global all_text2
    global all_Dots2 
    #Hides the selected pixel information
    canvas5.itemconfigure("selectedDots", state = "hidden")
    canvas5.itemconfigure("text", state = "hidden")
    canvas5.itemconfigure('random_selectedDots', state = 'hidden')
    canvas5.itemconfigure('random_text', state = 'hidden')
    if randDots2 != {} and textInfo4!= {}:
        for coordinates2 in randDots2:
            canvas5.delete(randDots2[coordinates2])
        for textID, (textX, textY) in textInfo4.items():
            canvas5.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if all_Dots2 != {} and all_text2 != {}:
        for coordinates2 in all_Dots2:
            canvas5.delete(all_Dots2[coordinates2])
        for textID, (textX, textY) in all_text2.items():
            canvas5.delete(textID)
    #This will plot the new dots2 and numbers on the image
    Count = 1
    for pixel in PixelList2:
        x = pixel[0] - min(cf2)
        y = pixel[1] - min(rf2)
        all_dot = canvas5.create_oval(x-5,y-5,x+5, y+5, fill = linecolor2, tag = "all_selectedDots")
        all_Dots2[(x,y)] = all_dot
        all_textID = canvas5.create_text(x+15, y, text = str(Count), fill = linecolor2, tag = "all_text2")
        all_text2[all_textID] = (x+15,y)
        Count += 1

    canvas_matplotlib.mpl_connect('button_press_event', clickAll_Selected2)
    canvas_matplotlib.mpl_connect('button_release_event', NotAll_Selected2)

# FINDS WHICH PIXEL CORRESPONDS TO THE LINE THE USER clicked2 ON
def whichAllPixel2():
    global all_Dots2 #contains the pixel coordinate and the dots2 drawn
    global selectedPixel2 #contains rgb values of the selected pixels
    global showTextAll2
    for keys in all_Dots2.keys():
        x,y = keys
        pixelkey = y,x
        dataPixel = ImageArr2[y][x]
        #trying to find the pixel clicked2 on and enlarge the dot and number
        if np.array_equal(dataPixel,selectedPixel2):
            global selectedDotAll2
            selectedDotAll2 = canvas5.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText_2.insert(tk.END, "Pixel coordinates2 = " + str(pixelkey) )


#CHECKS TO SEE IF YOU clicked2 ON A LINE IN THE ALL PIXEL PLOT
def clickAll_Selected2(event):
    global clickedAll2
    global allLinetracker2
    global allData2
    if event.inaxes == axs[3]: #checks to see if you clicked2 in the first plot
        for key, lineOb in allLinetracker2.items():
            contains, _ = lineOb.contains(event) #checks to see if an event happened near a line
            #contains is a true or false value based on if event happened on a line
            if contains:
                global selectedPixel2 # has rdg values of that line
                selectedPixel2 = allData2[key-1]
                whichAllPixel2() #function will draw a dot and print the pixel coordinate and rgb values
                break

#DELETES THE WHITE DOT AND NUMBER
def NotAll_Selected2(event):
    global selectedDotAll2
    global showTextAll2
    canvas5.delete(selectedDotAll2)
    # canvas5.delete(showTextAll2)
    PixelText_2.delete('1.0', tk.END)

#Gets the user input for every <blank> pixel to be graphed
def inputForStep2():
    global stepsize2
    stepsize2 = entry2_2.get()
    everyPixel2()

'''
THIS IS FOR SELECTING WHICH component2 TO PLOT
'''
# GET USER INPUT FOR WHICH component2 TO VISUALIZE
def get_componentNum2():
    global component2
    component2 = entry3_2.get()
    updatePlotforComp2()
    graphRandComp2()

# THIS IS TO GRAPH RANDOM PIXELS OF THAT component2
def graphRandComp2():
    global component2, randomNum2, randList2, randData2
    randList2 = [] # stores pixel coordinates2
    # clear2 dots2 of all pixels plot id its on the image
    for coordinates2 in all_Dots2:
        canvas5.delete(all_Dots2[coordinates2])
    for textID, (textX, textY) in all_text2.items():
        canvas5.delete(textID)
    #if there are random pixels already on the image, it will delete them
    if randDots2 != {} and textInfo4 != {}:
        for coordinates2 in randDots2:
            canvas5.delete(randDots2[coordinates2])
        for textID, (textX, textY) in textInfo4.items():
            canvas5.delete(textID)
    #gets pixels that correspond to the component2 number the user selects
    count = 1
    output = cv2.connectedComponentsWithStats(VNIR_Mask2.astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    compR, compC = np.where(labels == int(component2))
    global RGB_image2, tk_image2
    compCube = (wholeCube2[compR,compC])
    #plots those pixels spectra on the 1st graph
    for pixel in range (0,int(randomNum2)):
        pixelNum = np.random.randint(0,compCube.shape[0])
        data = compCube[pixelNum]
        randData2.append(data)
        randList2.append([compC[pixelNum], compR[pixelNum]])
        #gives you a line 2D object with properties of the line
        line, = axs[0].plot(index, data, color = linecolor2,alpha=alphaV)
       
        axs[0].annotate(str(count), (2.05,data[-1]), color = linecolor2) # adds the numbers to the end of the lines plotted
        randLineTracker2[count] = line
        count += 1
        fig.canvas.draw()
    # Updates the labeling on the average pixels plot 
    global mean_2
    mean_2.remove()
    axs[1].set_title('Average of component ' + str(component2))
    mean = np.mean(compCube,axis= 0)
    mean_2, = axs[1].plot(index, mean, color = linecolor2,alpha=alphaV)
    axs[1].set_ylim(0,1.2)
    axs[1].set_yticks(np.linspace(0,1,5))
    axs[1].set_xlabel('Spectral Band (nm)')
    axs[1].set_ylabel('Reflectance')
    fig.canvas.draw()

# CLEARS THE RANDOM PIXEL PLOT AND GRAPHS NEW PIXELS 
def updatePlotforComp2():
     global randLineTracker2
     for lines in randLineTracker2.values():
         lines.remove()
     randLineTracker2 = {}
     axs[0].set_title('Random Pixels')
     axs[0].set_ylim(0,1.2)
     axs[0].set_yticks(np.linspace(0,1,5))
     axs[0].set_xlabel('Spectral Band (nm)')
     axs[0].set_ylabel('Reflectance')
     global randData2
     randData2= []
     

#help_2messagebox that lists all of the feature information
def help_2():
    messagebox.showinfo('Guide' , "'Clear Selected Pixels' : Removes the dots that the user selected"+'\n' +  '\n' +"'Generate Random Pixels' : Plots new random pixel spectras"+'\n' +  '\n' +"'Show Random Pixels' : Only displays dots corresponding to the random pixel plot"
                        +'\n' +  '\n' + "'Show Selected Pixels' : Only displays dots for the Selected Pixels plot " +'\n' + '\n' + "'Show All Pixels': Only displays dots for every X Pixels"+'\n' + '\n' + "'Graph Every <blank> Pixels' : Plots every X number of pixels"
                        +'\n' + '\n' +  "'Enter Random Number of Pixels' : User can choose the number of pixels to be plotted in the Random Pixels plot"+'\n' + '\n' + "'Enter component Number': Select which component you want to generate random pixels and graph average spectra" +'\n' + '\n'  + "'Export' : Create an excel sheet with the selected pixel coordinates and HSI data"
                        +'\n' + '\n' +  "'Slider' : Chooses the pen's thickness" + '\n' + 'To select a region of pixels, right click and drag.' +'\n' + '\n' + "'Clear Image' : Remove Drawings and dots from the image"
                        +'\n' + '\n' + 'Left click on the image to graph the corresponding pixel spectra. You can click on any spectra in the 1st, 3rd, and 4th plots to get the pixel coordinates'
                         +'\n' + '\n' + 'Use the scroll bars to view the image if the image is too big to fit the canvas')
    

# Function that saves pixel coordinates2 and data in Excel
def export2():
    data = []
    for coordinates2 in imagedots2.keys():
        info = (coordinates2[1], coordinates2[0], *(wholeCube2[coordinates2[1],coordinates2[0]]))
        data.append(info)
    df = pd.DataFrame(data)
    columns = ['Row', 'Column', 'Data'] + [''] * (df.shape[1] - 3)
    df.columns = columns
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                             filetypes=[("Excel files", "*.xlsx"), 
                                                        ("All files", "*.*")])
    if file_path:
        # Save the DataFrame to an Excel file
        df.to_excel(file_path, index=False)

 #Clears image of all dots2 and pixel selections
def clear2(): 
    global RGB_image2,tk_image2
    RGB_image2 = Image.fromarray(realIm2) 
    tk_image2 = ImageTk.PhotoImage(RGB_image2)
    canvas5.create_image(0, 0, anchor=tk.NW, image=tk_image2) 

# THIS ALLOWS THE USER TO INTERACT WITH THE GRAPHS
canvas_matplotlib.mpl_connect('button_press_event', clickSelected2)
canvas_matplotlib.mpl_connect('button_release_event', NotSelected2)
canvas_matplotlib.mpl_connect('button_press_event', clickRand_Selected2)
canvas_matplotlib.mpl_connect('button_release_event', NotRand_Selected2)

# All of the Displays and their commands for the second hsi mage
frame_grid2 = tk.Frame(half, bg = 'white') #creates a grid to place all widgets
frame_grid2.grid(row=2, column=0, columnspan=4, sticky="ew")

button_0 = tk.Button(frame_grid2, text="Clear Selected Pixels", command = updatePlot2 )
button_0.grid(row = 0, column=0)

button2_2 = tk.Button(frame_grid2, text="Generate Random Pixels", command = updatePlotforRand2)
button2_2.grid(row = 0, column=1)

button4_2= tk.Button(frame_grid2, text="Show Random Pixels",command = showRandPixels2)
button4_2.grid(row = 0, column=2)

entry2_2 = tk.Entry(frame_grid2, width = 20)
entry2_2.grid(row = 1, column=0)

button3_2 = tk.Button(frame_grid2, text= "Graph Every <blank> Pixel", command = inputForStep2)
button3_2.grid(row = 1, column=1)

entry_2 = tk.Entry(frame_grid2, width=20)
entry_2.grid(row = 1, column=2)

button5_2 = tk.Button(frame_grid2, text="Enter Random Pixel Number", command=get_input2)
button5_2.grid(row = 1, column=3)

clearbutton_2 = tk.Button(frame_grid2, text="Clear Image", command=clear2)
clearbutton_2.grid(row = 1, column=4)

button6_2 = tk.Button(frame_grid2, text="Show Selected Pixels", command = showSelectedPixels2)
button6_2.grid(row = 0, column=3)

button7_2 = tk.Button(frame_grid2, text="Show All Pixels", command = showAllPixels2)
button7_2.grid(row = 0, column=4)

entry3_2 = tk.Entry(frame_grid2, width = 20)
entry3_2.grid(row = 2, column=0)

button8_2 = tk.Button(frame_grid2, text="Enter component Number", command = get_componentNum2)
button8_2.grid(row = 2, column=1)

export_button_2 = tk.Button(frame_grid2, text= "Export File", command = export2) 
export_button_2.grid(row=2, column=2) 

slider_2 = tk.Scale(frame_grid2, from_= 1, to_= 8, orient=tk.HORIZONTAL, command = penValue2)
slider_2.grid (row = 2, column=3)

help_2= tk.Button(frame_grid2, text="Help!", command = help_2)
help_2.grid(row = 2, column=4)

PixelText_2 = tk.Text(half, height = 1, width = 40)
PixelText_2.grid(row = 3, column=0)

root.mainloop()