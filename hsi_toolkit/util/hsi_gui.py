'''
Use this GUI if you want to visualize either VNIR-E data or SWIR data
You will need to edit the code from line 69 to line 87.
Give it a minute once you finished selecting the files
Click on the 'Help!' button and read the guide to understand the features
'''

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, Scrollbar
from PIL import Image, ImageTk, ImageDraw
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
canvas1 = tk.Canvas(canvasFrame, bg="white", width=frame1_width, height=frame1_height)
frame2 = tk.Frame(canvasFrame, bg="lightgrey", width=frame2_width, height=frame2_height)

# Place the frames in the divided parts
canvas1.grid(row=0, column=0, sticky="nsew") #nsew means frame expands to all sides
frame2.grid(row=0, column=1, sticky="nsew")


canvasFrame.update_idletasks()  # Ensure the canvas dimensions are updated
canvas.config(scrollregion=canvas.bbox("all"))

'''
USER NEEDS TO CHANGE: FILE PATH LOCATION AND HOW THEY UPLOAD THEIR DATA
'''
# Opens Image Path. This is to get the rgb image and display it on the canvas 
imagePath = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Image", filetypes = (('png files', '*.png'),('all files', '*.*'),))

# Opens data paths
data_hdr = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data.hdr", filetypes = (('hdr files', '*.hdr'),('all files', '*.*'),))
data = filedialog.askopenfilename(initialdir = '/anika/HSI image', title = "Data", filetypes = (('all files', '*.*'),))
hsi_ref = envi.open(data_hdr,data) # change how you upload your data if necessary
hsi_np = np.copy(hsi_ref.asarray())
wholeCube = (hsi_np)

#Creates image and data cube
image = Image.open(imagePath)
tk_image = ImageTk.PhotoImage(image)
ImageArr = np.array(wholeCube)

'''
USER SHOULD NOT NEED TO CHANGE ANY CODE AFTER THIS POINT
'''

#determines x axis range based on if swir data or vnir data was chosen
index = np.linspace(900,2500,wholeCube.shape[2])
if wholeCube.shape[-1] == 372:
    index = np.linspace(400,1000,wholeCube.shape[2])

# Creates canvas3 to create a scollable area for image and creates canvas 2 to hold the image
canvas3 = tk.Canvas(canvas1, width = frame1_width, height = 730)
canvas3.grid(row = 0, column= 0, sticky = 'nsew')
canvas2 = tk.Canvas(canvas3, width = image.width, height = image.height)
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
canvas1.config(scrollregion=(0, 0,frame1_width, 730))

# Create a Matplotlib figure and display a plot in the second frame
fig, axs = plt.subplots(4, 1, figsize=(frame2_width/100  , frame2_height/70 - 10))
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
THIS IS ALL THE CODE FOR THE AVERAGE PIXEL GRAPH (SECOND PLOT)
'''
#Plots the mean spectra
mean2 = np.mean(wholeCube,axis= (0,1))
axs[1].plot(index, mean2)

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
    global ColorList
    ColorList = [] # keeps track of the line colors
    count = 1
    for pixel in range (0,int(randomNum)):
        pixelW = np.random.randint(0,ImageShape[0])
        pixelH = np.random.randint(0,ImageShape[1])
        data = ImageArr[pixelW, pixelH]
        randData.append(data)
        randList.append([pixelH,pixelW])
        #gives you a line 2D object with properties of the line
        line, = axs[0].plot(index, data)
        color = line.get_color()
        axs[0].annotate(str(count), (2.05,data[-1]), color = color) # adds the numbers to the end of the lines plotted
        randLineTracker[count] = line
        count += 1
        ColorList.append(color)
        fig.canvas.draw()

# Calls the function intially
randomPixels(wholeCube.shape)

# CLEARS THE RANDOM PIXEL PLOT AND GRAPHS NEW PIXELS 
def updatePlotforRand():
     axs[0].clear()
     axs[0].set_title('Random Pixels')
     axs[0].set_ylim(0,1.2)
     axs[0].set_yticks(np.linspace(0,1,5))
     axs[0].set_xlabel('Spectral Band (nm)')
     axs[0].set_ylabel('Reflectance')
     global randData
     randData = []
     global randLineTracker
     randLineTracker = {}
     fig.canvas.draw() 
     randomPixels(wholeCube.shape)  

# These 2 dicts hold the information on dots and numbers plotted on image
randDots = {} 
textInfo2 = {}

# GRAPHS THE DOTS AND NUMBERS OFTHE RANDOM PIXELS ON THE IMAGE
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
        x = pixel[0]
        y = pixel[1]
        rand_dot = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = ColorList[Count-1], tag = "random_selectedDots")
        randDots[(x,y)] = rand_dot
        rand_textID = canvas2.create_text(x+15, y, text = str(Count), fill = ColorList[Count-1], tag = "random_text")
        textInfo2[rand_textID] = (x+15,y)
        Count += 1

# FINDS WHICH PIXEL CORRESPONDS TO THE LINE THE USER CLICKED ON
def whichRandomPixel():
    global randDots #contains the pixel coordinate and the dots drawn
    global yCoords2 #contains rgb values of the selected pixels

    for coordinates in all_Dots:
        canvas2.delete(all_Dots[coordinates])
    for textID, (textX, textY) in all_text.items():
        canvas2.delete(textID)
    for keys in randDots.keys():
        x,y = keys
        pixelkey = y,x
        dataPixel = ImageArr[y][x]
        #trying to find the pixel clicked on and enlarge the dot and number
        if np.array_equal(dataPixel,yCoords2):
            global selectedDot2
            selectedDot2 = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = "white")
            PixelText.insert(tk.END, "Pixel Coordinates = " + str(pixelkey) )

#CHECKS TO SEE IF YOU CLICKED ON A LINE IN THE RANDOM PIXEL PLOT
def clickRand_Selected(event):
    # global clicked
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
    global showText2
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
    global color
    line, = axs[2].plot(index,data)
    lineTracker[dotCount] = line
    color = line.get_color()
    axs[2].annotate(str(dotCount), (2.05,data[-1]), color = color)
    fig.canvas.draw()

#THIS FUNCTION CLEARS THE SELECTED PIXEL PLOTS
def updateSelected4Dots(): 
     axs[2].clear()
     global ycoordList
     ycoordList = []
     global lineTracker
     lineTracker = {}
     #relabeling plot
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
        line, = axs[2].plot(index,data)
        lineTracker[dotCount] = line
        color = line.get_color()
        axs[2].annotate(str(dotCount), (2.05,data[-1]), color = color)
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
        dotCount += -1
        del imagedots[coordinates]
        listRedraw = list(imagedots.keys())
        redraw_pixel_rgb(listRedraw) #update graph if user unselects a pixel 
    else:
        global color
        dot = canvas2.create_oval(x-3,y-3,x+2, y+2, fill = color, tag = "selectedDots")
        imagedots[coordinates] = dot
        textID = canvas2.create_text(x+10, y, text = str(dotCount), fill = color, tag = "text")
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
            PixelText.insert(tk.END, "Pixel Coordinates = " + str(pixelkey))

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
   
#THIS CLEARS THE SELECTED DOTS GRAPH AND EMPTIES ALL LISTS/DICTS
def updatePlot():
     #updates the plots labeling
     axs[2].clear()
     axs[2].set_title('Selected Pixels')
     axs[2].set_ylim(0,1.2)
     axs[2].set_yticks(np.linspace(0,1,5))
     axs[2].set_xlabel('Spectral Band (nm)')
     axs[2].set_ylabel('Reflectance')
    # Redraws the image so its empty
     global image,tk_image
     image = Image.open(imagePath)
     tk_image = ImageTk.PhotoImage(image)
     canvas2.create_image(0, 0, anchor=tk.NW, image=tk_image) 
     # empties all lists and dicts that keep track of selected pixels
     global ycoordList
     global coordinates
     coordinates = []
     ycoordList = []
     global lineTracker
     lineTracker = {}
     global dotCount
     dotCount = 1
     fig.canvas.draw()

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
pen_size = 1 # initial pen value

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

colors = []  #keeps track of line colors for pixels the user drew on
#Plots the spectra for pixels that were drew over
def graphDraggedPixel(list_to_redraw):
    updateSelected4Dots()
    global dotCount
    dotCount = 1
    for coordinate in list_to_redraw:
        x,y = coordinate
        data = ImageArr[y,x]
        ycoordList.append(data)
        line, = axs[2].plot(index,data)
        lineTracker[dotCount] = line
        color = line.get_color()
        colors.append(color)
        axs[2].annotate(str(dotCount), (2.05,data[-1]), color = color)
        dotCount += 1
        fig.canvas.draw()

# Creates dots on the pixels that the user drew over
def on_button_release(event):
    global last_x,last_y
    global dotCount
    last_x, last_y = None, None
    global coordinates
    graphDraggedPixel(coordinates)
    colorCount = 0
    for x,y in coordinates:
        if (x,y) not in imagedots:
            coord = (x,y)
            dot = canvas2.create_oval(x-3,y-3,x+2, y+2, fill = colors[colorCount], tag = "selectedDots")
            imagedots[coord] = dot
            textID = canvas2.create_text(x+10, y, text = str(dotCount), fill = colors[colorCount], tag = "text")
            textInfo[textID] = (x+10,y)
            dotCount += 1
            colorCount+=1

# Allows the user to draw on the image  
def draw_on_image( x, y):
        global tk_image, image
        global last_x,last_y, pen_size
        # Draw on the image using PIL
        draw = ImageDraw.Draw(image)
        draw.line([last_x, last_y, x, y], fill=pen_color, width=pen_size)
        # Update the canvas
        tk_image = ImageTk.PhotoImage(image)
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
    global stepsize, PixelList, alldata, allLinetracker, colorTracker
    alldata = []
    allLinetracker = {}
    colorTracker = []
    PixelList= []
    all_count = 1
    #resets the 4th graph
    axs[3].clear()
    axs[3].set_title('All Pixels')
    axs[3].set_ylim(0,1.2)
    axs[3].set_yticks(np.linspace(0,1,5))
    axs[3].set_xlabel('Spectral Band (nm)')
    axs[3].set_ylabel('Reflectance')
    fig.canvas.draw()

    #graphs every <blank> pixels
    for pixelR in range(0, ImageArr.shape[0],int(int(stepsize)/2)):
        for pixelC in range(0,ImageArr.shape[1],int(int(stepsize)/2)):
            PixelList.append([pixelC,pixelR])
            rgb = ImageArr[pixelR,pixelC]
            alldata.append(rgb)
            all_line, = axs[3].plot(index, rgb)
            allLinetracker[all_count] = all_line
            all_color = all_line.get_color()
            colorTracker.append(all_color)
            axs[3].annotate(str(all_count), (2.05,rgb[-1]), color = all_color) # adds the numbers to the end of the lines plotted
            all_count += 1
    fig.canvas.draw()

# These 2 dicts hold the information on dots and numbers plotted on image
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
        x = pixel[0]
        y = pixel[1]
        all_dot = canvas2.create_oval(x-5,y-5,x+5, y+5, fill = colorTracker[Count-1], tag = "all_selectedDots")
        all_Dots[(x,y)] = all_dot
        all_textID = canvas2.create_text(x+15, y, text = str(Count), fill = colorTracker[Count-1], tag = "all_text")
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
    global alldata
    if event.inaxes == axs[3]: #checks to see if you clicked in the first plot
        for key, lineOb in allLinetracker.items():
            contains, _ = lineOb.contains(event) #checks to see if an event happened near a line
            #contains is a true or false value based on if event happened on a line
            if contains:
                global selectedPixel # has rdg values of that line
                selectedPixel = alldata[key-1]
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

#help messagebox that lists all of the feature information
def help():
    messagebox.showinfo('Guide' , "'Clear Selected Pixels' : Removes the dots that the user selected"+'\n' +  '\n' +"'Generate Random Pixels' : Plots new random pixel spectras"+'\n' +  '\n' +"'Show Random Pixels' : Only displays dots corresponding to the random pixel plot"
                        +'\n' +  '\n' + "'Show Selected Pixels' : Only displays dots for the Selected Pixels plot " +'\n' + '\n' + "'Show All Pixels': Only displays dots for every X Pixels"+'\n' + '\n' + "'Graph Every <blank> Pixels' : Plots every X number of pixels"
                        +'\n' + '\n' +  "'Enter Random Number of Pixels' : User can choose the number of pixels to be plotted in the Random Pixels plot"+'\n' + '\n'  + "'Export' : Create an excel sheet with the selected pixel coordinates and HSI data"+'\n' + '\n'  + 
                       "'Slider' : Chooses the pen's thickness" + '\n' + 'To select a region of pixels, right click and drag.' +'\n' + '\n' + "'Clear Image' : Remove Drawings and dots from the image"
                       +'\n' + '\n' + 'Left click on the image to graph the corresponding pixel spectra. You can click on any spectra in the 1st, 3rd, and 4th plots to get the pixel coordinates'
                        +'\n' + '\n' + 'Use the scroll bars to view the image if the image is too big to fit the canvas')

# THIS ALLOWS THE USER TO INTERACT WITH THE GRAPHS
canvas_matplotlib.mpl_connect('button_press_event', clickSelected)
canvas_matplotlib.mpl_connect('button_release_event', NotSelected)
canvas_matplotlib.mpl_connect('button_press_event', clickRand_Selected)
canvas_matplotlib.mpl_connect('button_release_event', NotRand_Selected)

# Function that saves pixel coordinates and data in Excel
def export():
    data = []
    for coordinates in imagedots.keys():
        pixelData = wholeCube[coordinates[1], coordinates[0], :]
        info = (coordinates[1], coordinates[0], *pixelData)
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
    global image,tk_image
    image = Image.open(imagePath)
    tk_image = ImageTk.PhotoImage(image)
    canvas2.create_image(0, 0, anchor=tk.NW, image=tk_image) 

# All of the Displays and their commands
frame_grid = tk.Frame(canvas1, bg = 'white')
frame_grid.grid(row=2, column=0, columnspan=4, sticky="ew")

button = tk.Button(frame_grid, text="Clear Selected Pixels", command = updatePlot )
button.grid(row = 0, column=0, pady = 1)

button2 = tk.Button(frame_grid, text="Generate Random Pixels", command = updatePlotforRand)
button2.grid(row = 0, column=1, pady = 1)

button4 = tk.Button(frame_grid, text="Show Random Pixels",command = showRandPixels)
button4.grid(row = 0, column=2, pady = 1)

entry2 = tk.Entry(frame_grid, width = 20)
entry2.grid(row = 1, column=0, pady = 5)

button3 = tk.Button(frame_grid, text= "Graph Every <blank> Pixel", command = inputForStep)
button3.grid(row = 1, column=1, pady = 1)

export_button = tk.Button(frame_grid, text= "Export File", command = export) 
export_button.grid(row=2, column=0) 

entry = tk.Entry(frame_grid, width=20)
entry.grid(row = 1, column=2, pady = 5)

slider = tk.Scale(frame_grid, from_= 1, to_= 8, orient=tk.HORIZONTAL, command = penValue)
slider.grid (row = 2, column=1, pady = 5)

clearbutton = tk.Button(frame_grid, text="Clear Image", command=clear)
clearbutton.grid(row = 1, column=4, pady = 5)

button5 = tk.Button(frame_grid, text="Enter Random Pixel Number", command=get_input)
button5.grid(row = 1, column=3, pady = 5)

button6 = tk.Button(frame_grid, text="Show Selected Pixels", command = showSelectedPixels)
button6.grid(row = 0, column=3, pady = 5)

button7 = tk.Button(frame_grid, text="Show All Pixels", command = showAllPixels)
button7.grid(row = 0, column=4, pady = 5)

help = tk.Button(frame_grid, text="Help!", command = help)
help.grid(row = 2, column=2, pady = 5)

PixelText = tk.Text(canvas1, height = 1, width = 40)
PixelText.grid(row = 3, column=0)

root.mainloop()

