from Tkinter import *
from PIL import ImageTk, Image
from random import *
from DCGAN import DCGAN
from utils import *
from embedding import tools
import scipy.misc
import numpy as np




path="output.png"
window_height = 540
window_width = 720
iconsize = 64
finalsize = 128

class GUI:
	def __init__(self, master, dcgan):
		self.embedding_model = tools.load_model()
		self.dcgan = dcgan
		self.master = master


		self.master.title("Icarusization@github")

		self.panedwindow = PanedWindow(master)
		self.panedwindow.pack(fill=BOTH, expand=1)

		self.left = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.left, width=window_width/3, height=window_height)

		self.middle = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.middle, width=window_width/3, height=window_height)

		self.right = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.right, width=window_width/3, height=window_height)

		self.textField = Text(self.left, height=window_height/3, width=window_width/3)
		self.textField.insert(END, "Please enter here")
		self.left.add(self.textField, height=window_height/3)

		self.textButtons = PanedWindow(self.left, orient=HORIZONTAL)
		self.left.add(self.textButtons, height=window_height/6)
		self.textButtonrun = Button(self.textButtons, text="Run", command=self.textRun)
		self.textButtonrandom = Button(self.textButtons, text="Random", command=self.textRandom)
		self.textButtonreset = Button(self.textButtons, text="Reset", command=self.textReset)
		self.textButtons.add(self.textButtonrun, width=window_width/9, height=window_height/6)
		self.textButtons.add(self.textButtonrandom, width=window_width/9, height=window_height/6)
		self.textButtons.add(self.textButtonreset, width=window_width/9, height=window_height/6)

		self.s1 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL)
		self.left.add(self.s1, height=window_height/9)

		self.s2 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL)
		self.left.add(self.s2, height=window_height/9)

		self.s3 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL)
		self.left.add(self.s3, height=window_height/9)


		self.scaleButtons = PanedWindow(self.left, orient=HORIZONTAL)
		self.left.add(self.scaleButtons, height=window_height/6)
		self.scaleButtonrun = Button(self.scaleButtons, text="Run", command=self.scaleRun)
		self.scaleButtonrandom = Button(self.scaleButtons, text="Random", command=self.scaleRandom)
		self.scaleButtonreset = Button(self.scaleButtons, text="Reset", command=self.scaleReset)
		self.scaleButtons.add(self.scaleButtonrun, width=window_width/9, height=window_height/6)
		self.scaleButtons.add(self.scaleButtonrandom, width=window_width/9, height=window_height/6)
		self.scaleButtons.add(self.scaleButtonreset, width=window_width/9, height=window_height/6)


		self.textImageField = Canvas(self.middle, width=window_width/2, height=window_height/2)
		self.middle.add(self.textImageField, height=window_height/2, width=window_width/3)
		self.textImageField.bind("<Button-1>", self.text_clicked)
		self.textImagesIcon=[]
		
		self.scaleImageField = Canvas(self.middle, width=window_width/2, height=window_height/2)
		self.middle.add(self.scaleImageField, height=window_height/2, width=window_width/3)
		self.scaleImageField.bind("<Button-1>", self.scale_clicked)
		self.scaleImagesIcon=[]
		
		self.mergedimage = Label(self.right)
		self.right.add(self.mergedimage, height=window_height*2/3, width=window_width/3)

		




		

		self.projectButtons = PanedWindow(self.right, orient=HORIZONTAL)
		self.right.add(self.projectButtons)
		self.projectButtonrun = Button(self.projectButtons, text="Run", command=self.projectRun)
		self.projectButtonrandom = Button(self.projectButtons, text="Random", command=self.projectRandom)
		self.projectClose = Button(self.projectButtons, text="Close", command=master.quit)
		self.projectButtons.add(self.projectButtonrun, width=window_width/9, height=window_height/6)
		self.projectButtons.add(self.projectButtonrandom, width=window_width/9, height=window_height/6)
		self.projectButtons.add(self.projectClose, width=window_width/9, height=window_height/6)
		
	def textRun(self):
		text = self.textField.get("1.0",END)
		embeddings=tools.encode_sentences(self.embedding_model,X=[text], verbose=False)
		self.textImagesIcon=[]
		imgs=np.clip(128*(self.dcgan.display(embeddings)+1),0,255)
		imgs=np.uint8(imgs)
		self.textImages = imgs
		for i in range(9):
			img=ImageTk.PhotoImage(Image.fromarray(imgs[i]).resize((iconsize,iconsize)))
			self.textImagesIcon.append(img)
			self.textImageField.create_image(window_width*((i%3)/9.0+1.0/18.0),window_height*((i/3)/6.0+1.0/12.0),image=self.textImagesIcon[i])
		

		#self.bigimg = ImageTk.PhotoImage(Image.fromarray(imgs[0]).resize((finalsize,finalsize)))
		#self.mergedimage.config(image=self.bigimg)


	def textReset(self):
		pass

	def textRandom(self):
		pass

	def scaleRun(self):
		pass

	def scaleReset(self):
		pass

	def scaleRandom(self):
		pass
	
	def projectRandom(self):
		pass
		
	def projectRun(self):
		pass	

	def text_clicked(self, event):
		x, y = event.x, event.y
		figureid=int(6*y/window_height)*3 + int(9*x/window_width)
		self.bigimg=ImageTk.PhotoImage(Image.fromarray(self.textImages[figureid]).resize((finalsize,finalsize)))
		self.mergedimage.config(image=self.bigimg)

	def scale_clicked(self, event):
		pass

if __name__=='__main__':
	root = Tk()
	my_gui = GUI(root)

	root.mainloop()