from Tkinter import *
from PIL import ImageTk, Image
from VSF import VerticalScrolledFrame
from random import *
from DCGAN import DCGAN
from utils import *
from embedding import tools
import scipy.misc
import numpy as np




path="output.png"
window_height = 450
window_width = 900
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




		self.img = ImageTk.PhotoImage(Image.open(path).resize((64,64)))
		self.bigimg = ImageTk.PhotoImage(Image.open(path).resize((finalsize,finalsize)))


		self.textImageField = VerticalScrolledFrame(self.middle)
		self.middle.add(self.textImageField, height=window_height/2, width=window_width/3)
		self.textImageLabels=[]
		
		self.scaleImageField = VerticalScrolledFrame(self.middle)
		self.middle.add(self.scaleImageField, height=window_height/2, width=window_width/3)

		
		self.mergedimage = Label(self.right, image=self.bigimg)
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
		for label in self.textImageLabels:
			label.pack_forget()
		text = self.textField.get("1.0",END)
		embeddings=tools.encode_sentences(self.embedding_model,X=[text], verbose=False)
		self.textImages=[]
		imgs=np.clip(128*(self.dcgan.display(embeddings)+1),0,255)
		imgs=np.uint8(imgs)
		for i in range(10):
			img=ImageTk.PhotoImage(Image.fromarray(imgs[i]).resize((iconsize,iconsize)))
			self.textImages.append(img)
		
		for i in range(10):
			self.textImageLabels.append(Label(self.textImageField.interior, image=self.textImages[i]))
			self.textImageLabels[-1].pack()

		self.bigimg = ImageTk.PhotoImage(Image.fromarray(imgs[0]).resize((finalsize,finalsize)))
		self.mergedimage.config(image=self.bigimg)


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


if __name__=='__main__':
	root = Tk()
	my_gui = GUI(root)

	root.mainloop()