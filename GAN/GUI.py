from Tkinter import *
from PIL import ImageTk, Image
from random import *
from WGAN import WGAN
from utils import *
import scipy.misc
import numpy as np
import tensorflow as tf

path="input.jpg"
zmapwidth=20
zmapcount=10
MARGIN=5
displaysize=128
WIDTH=zmapwidth*zmapcount
HEIGHT=zmapwidth*zmapcount




class GUI:
    def __init__(self, master, wgan=None):
        self.wgan = wgan
        self.path = path
        self.col = int(-1)
        self.row = int(-1)
        self.zdata = np.zeros((zmapcount,zmapcount),dtype=np.int32)

        self.master = master
        master.title("Icarusization@github")

        self.label = Label(master, text="Try to create a figure!")
        self.label.pack()

        
        self.display = Canvas(master, width=displaysize, height=displaysize)
        self.display.pack()

        self.zmap = Canvas(master, width=WIDTH+5, height=HEIGHT+5)
        self.zmap.pack()
        self.drawzmap()

        self.zmap.bind("<Button-1>", self.cell_clicked)

        self.slider = Scale(master, from_=0, to=99, length=300, tickinterval=99, orient=HORIZONTAL, command=self.slided)
        self.slider.pack()

        self.run_button = Button(master, text="Run", command=self.run, highlightcolor="red")
        self.run_button.pack()
        
        self.shuffle_button = Button(master, text="Shuffle", command=self.shuffle)
        self.shuffle_button.pack()
        
        self.reset_button = Button(master, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()


    def reset(self):
        self.col = int(-1)
        self.row = int(-1)
        self.zdata = np.zeros((zmapcount,zmapcount),dtype=np.int32)
        self.slider.set(0)
        self.drawzmap()
        

    def run(self):
        z = (self.zdata.reshape((1,100))-49.5)/49.5
        img = self.wgan.display(self.path, z)
        img = np.clip(128*(img+1),0,255)
        img = np.uint8(img)
        scipy.misc.imsave("output.png", img)
        array = Image.fromarray(img).resize((displaysize,displaysize))
        self.img = ImageTk.PhotoImage(array)
        self.display.create_image(displaysize/2,displaysize/2,image=self.img)
        #self.display.pack()

    def shuffle(self):
        self.col = int(-1)
        self.row = int(-1)
        for i in xrange(zmapcount):
            for j in xrange(zmapcount):
                self.zdata[i][j] = randint(0,99)
        self.drawzmap()

    def slided(self, val):
        col, row = self.col, self.row
        if col!=-1:
            self.zdata[row][col]=int(val)
            color = "gray"+str(99-int(val))
            x0 = MARGIN + col * zmapwidth
            y0 = MARGIN + row * zmapwidth 
            x1 = MARGIN + (col+1) * zmapwidth
            y1 = MARGIN + (row+1) * zmapwidth 
            self.zmap.create_rectangle(x0, y0, x1, y1, fill = color)
        self.red()

    def drawzmap(self):
        for i in xrange(zmapcount+1):
            color = "black" 

            x0 = MARGIN + i * zmapwidth 
            y0 = MARGIN 
            x1 = MARGIN + i * zmapwidth 
            y1 = HEIGHT + MARGIN  
            self.zmap.create_line(x0, y0, x1, y1, fill=color)

            x0 = MARGIN 
            y0 = MARGIN + i * zmapwidth 
            x1 = WIDTH + MARGIN 
            y1 = MARGIN + i * zmapwidth 
            self.zmap.create_line(x0, y0, x1, y1, fill=color)
        
        for i in xrange(zmapcount):
            for j in xrange(zmapcount):
                color = "gray"+str(99-self.zdata[i][j])
                x0 = MARGIN + j * zmapwidth
                y0 = MARGIN + i * zmapwidth 
                x1 = MARGIN + (j+1) * zmapwidth
                y1 = MARGIN + (i+1) * zmapwidth 
                self.zmap.create_rectangle(x0, y0, x1, y1, fill = color)

        self.red()

        

    def cell_clicked(self, event):
        x, y = event.x, event.y
        if (MARGIN < x < WIDTH - MARGIN and MARGIN < y < HEIGHT - MARGIN):
            #self.zmap.focus_set()

            # get row and col numbers from x,y coordinates
            self.row, self.col = int((y - MARGIN) / zmapwidth), int((x - MARGIN) / zmapwidth)
            self.slider.set(self.zdata[self.row][self.col])
            self.drawzmap()

    def red(self):
        col, row = self.col, self.row
        if col!=-1:
            for i in xrange(2):
                color = "red" 

                x0 = MARGIN + (i+col) * zmapwidth 
                y0 = MARGIN + row * zmapwidth
                x1 = MARGIN + (i+col) * zmapwidth 
                y1 = MARGIN + (row+1) * zmapwidth
                self.zmap.create_line(x0, y0, x1, y1, fill=color)

                x0 = MARGIN + col * zmapwidth
                y0 = MARGIN + (i+row) * zmapwidth 
                x1 = MARGIN + (col+1) * zmapwidth
                y1 = MARGIN + (i+row) * zmapwidth 
                self.zmap.create_line(x0, y0, x1, y1, fill=color)

            
        
if __name__=='__main__':
    
    root = Tk()
    img = ImageTk.PhotoImage(Image.open(path).resize((64,64)))
    
    my_gui = GUI(root, img)

    root.mainloop()