from Tkinter import *
from PIL import ImageTk, Image
from random import randint
from DCGAN import DCGAN
from utils import *
from embedding import tools
import scipy.misc
import numpy as np
import os




path="output.png"
unit_height = 270
unit_width = 270
bar_height = 40
iconsize = 64
finalsize = 128

zmapwidth=20
zmapcount=10
MARGIN=5
displaysize=128
WIDTH=zmapwidth*zmapcount
HEIGHT=zmapwidth*zmapcount

class GUI:
	def __init__(self, master, dcgan=None, dcgan2=None):
		self.embedding_model = tools.load_model()
		self.dcgan = dcgan
		self.dcgan2 = dcgan2
		self.master = master


		self.zdata = np.zeros(90,dtype=np.int32)

		self.master.title("The birth of Icarus.")

		self.panedwindow = PanedWindow(master)
		self.panedwindow.pack(fill=BOTH, expand=1)

		self.left = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.left, width=unit_width, height=2*unit_height+4*bar_height+20)

		self.middle = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.middle, width=unit_width, height=2*unit_height+4*bar_height+20)

		self.right = PanedWindow(self.panedwindow, orient=VERTICAL)
		self.panedwindow.add(self.right, width=unit_width, height=2*unit_height+4*bar_height+20)

		self.barleftup = Label(self.left, bg='gray', text="Enter a sentense as content.")
		self.left.add(self.barleftup, height=bar_height)

		self.textField = Text(self.left, height=unit_height, width=unit_width)
		self.textField.insert(END, "Please enter here")
		self.left.add(self.textField, height=unit_height)

		self.textButtons = PanedWindow(self.left, orient=HORIZONTAL)
		self.left.add(self.textButtons, height=bar_height)
		self.textButtonrun = Button(self.textButtons, text="Run", command=self.textRun)
		self.textButtonrandom = Button(self.textButtons, text="Random", command=self.textRandom)
		self.textButtonreset = Button(self.textButtons, text="Reset", command=self.textReset)
		self.textButtons.add(self.textButtonrun, width=unit_width/3, height=bar_height)
		self.textButtons.add(self.textButtonrandom, width=unit_width/3, height=bar_height)
		self.textButtons.add(self.textButtonreset, width=unit_width/3, height=bar_height)

		self.barleftdown = Label(self.left, bg='gray50', text="Generate a style.")
		self.left.add(self.barleftdown, height=bar_height)
		
		self.s1 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL, command=self.slided1)
		self.left.add(self.s1, height=unit_height/3)

		self.s2 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL, command=self.slided2)
		self.left.add(self.s2, height=unit_height/3)

		self.s3 = Scale(self.left, from_=0, width=10, to=99, length=30, tickinterval=99, orient=HORIZONTAL, command=self.slided3)
		self.left.add(self.s3, height=unit_height/3)
		

		self.scaleButtons = PanedWindow(self.left, orient=HORIZONTAL)
		self.left.add(self.scaleButtons, height=bar_height)
		self.scaleButtonrun = Button(self.scaleButtons, text="Run", command=self.scaleRun)
		self.scaleButtonrandom = Button(self.scaleButtons, text="Random", command=self.scaleRandom)
		self.scaleButtonreset = Button(self.scaleButtons, text="Reset", command=self.scaleReset)
		self.scaleButtons.add(self.scaleButtonrun, width=unit_width/3, height=bar_height)
		self.scaleButtons.add(self.scaleButtonrandom, width=unit_width/3, height=bar_height)
		self.scaleButtons.add(self.scaleButtonreset, width=unit_width/3, height=bar_height)

		self.barmiddleup = Label(self.middle, bg='gray', text="Select the content.")
		self.middle.add(self.barmiddleup, height=bar_height)

		self.textImageField = Canvas(self.middle, width=unit_width, height=unit_height+bar_height)
		self.middle.add(self.textImageField, width=unit_width, height=unit_height)
		self.blank1 = Label(self.middle)
		self.middle.add(self.blank1, width=unit_width, height=bar_height)
		self.textImageField.bind("<Button-1>", self.text_clicked)
		self.textImagesIcon=[]
		
		self.barmiddledown = Label(self.middle, bg='gray50', text="Select the style.")
		self.middle.add(self.barmiddledown, height=bar_height)

		self.scaleImageField = Canvas(self.middle, width=unit_width, height=unit_height+bar_height)
		self.middle.add(self.scaleImageField, width=unit_width, height=unit_height+bar_height)
		self.blank2 = Label(self.middle)
		self.middle.add(self.blank2, width=unit_width, height=bar_height)
		self.scaleImageField.bind("<Button-1>", self.scale_clicked)
		self.scaleImagesIcon=[]
		
		self.logolabel = Label(self.right, bg='pink', text="Final Result.")
		self.right.add(self.logolabel, height=bar_height, width=unit_width)

		self.mergedimage = Label(self.right)
		self.right.add(self.mergedimage, height=unit_height, width=unit_width)

		self.blank3 = Label(self.right)
		self.right.add(self.blank3, width=unit_width, height=bar_height)

		self.rightlabel = Label(self.right, bg='pink', text="The Chosen Content and Style.")
		self.right.add(self.rightlabel, height=bar_height)

		self.selectedimages = PanedWindow(self.right, orient = HORIZONTAL)
		self.right.add(self.selectedimages, width=unit_width, height=unit_height)
		self.selectedtext = Label(self.right)
		self.selectedstyle = Label(self.right)
		self.selectedimages.add(self.selectedtext, height=unit_height/2, width=unit_width/2)
		self.selectedimages.add(self.selectedstyle, height=unit_height/2, width=unit_width/2)


		

		self.projectButtons = PanedWindow(self.right, orient=HORIZONTAL)
		self.right.add(self.projectButtons, width=unit_width/3, height=bar_height)
		self.projectButtonrun = Button(self.projectButtons, text="Run", command=self.projectRun)
		self.projectClose = Button(self.projectButtons, text="Close", command=master.quit)
		self.projectButtons.add(self.projectButtonrun, width=unit_width/2, height=bar_height)
		self.projectButtons.add(self.projectClose, width=unit_width/2, height=bar_height)



		#self.memberinfo = Label(self.right, text="icarusization@Github\n shyay1013@gmail.com\n")
		#self.right.add(self.memberinfo,height=unit_height/2 - bar_height, width=unit_width)

		self.intro = Label(self.right, bg='pink', text="By Icarus-A Student Team That\n Deploy the Power of AI in Arts.")
		self.right.add(self.intro, height=bar_height, width=unit_width)
		
	def textRun(self):
		text = self.textField.get("1.0",END)
		embeddings=tools.encode_sentences(self.embedding_model,X=[text], verbose=False)
		self.textImagesIcon=[]
		imgs=np.clip(128*(self.dcgan.display(embeddings=embeddings)+1),0,255)
		imgs=np.uint8(imgs)
		self.textImages = imgs
		for i in range(9):
			img=ImageTk.PhotoImage(Image.fromarray(imgs[i]).resize((iconsize,iconsize)))
			self.textImagesIcon.append(img)
			self.textImageField.create_image(unit_width*((i%3)/3.0+1.0/6.0),unit_height*((i/3)/3.0+1.0/6.0),image=self.textImagesIcon[i])
		

		#self.bigimg = ImageTk.PhotoImage(Image.fromarray(imgs[0]).resize((finalsize,finalsize)))
		#self.mergedimage.config(image=self.bigimg)


	def textReset(self):
		self.textField.delete("1.0",END)

	def textRandom(self):
		pass

	def scaleRun(self):
		z = (self.zdata-49.5)/49.5
		self.scaleImagesIcon=[]
		imgs=np.clip(128*(self.dcgan2.display(z=z)+1),0,255)
		imgs=np.uint8(imgs)
		self.scaleImages = imgs
		for i in range(9):
			img=ImageTk.PhotoImage(Image.fromarray(imgs[i]).resize((iconsize,iconsize)))
			self.scaleImagesIcon.append(img)
			self.scaleImageField.create_image(unit_width*((i%3)/3.0+1.0/6.0),unit_height*((i/3)/3.0+1.0/6.0),image=self.scaleImagesIcon[i])

	def scaleReset(self):
		self.zdata = np.zeros(1024,dtype=np.int32)
		self.s1.set(0)
		self.s2.set(0)
		self.s3.set(0)
		self.scaleRun()

	def scaleRandom(self):
		r=randint(0,99)
		self.s1.set(r)
		for i in xrange(0,90,3):
			self.zdata[i] = r
		r=randint(0,99)
		self.s2.set(r)
		for i in xrange(1,90,3):
			self.zdata[i] = r
		r=randint(0,99)
		self.s3.set(r)
		for i in xrange(2,90,3):
			self.zdata[i] = r
		self.scaleRun()
	
		
	def projectRun(self):
		os.system('python neural_style.py --content content.jpg --styles style.jpg --output output.jpg')
		self.bigimg=ImageTk.PhotoImage(Image.open("output.jpg").resize((finalsize,finalsize)))
		self.mergedimage.config(image=self.bigimg)	

	def slided1(self, val):
		for i in xrange(0,90,3):
			self.zdata[i] = int(val)
	def slided2(self, val):
		for i in xrange(1,90,3):
			self.zdata[i] = int(val)
	def slided3(self, val):
		for i in xrange(2,90,3):
			self.zdata[i] = int(val)

	def text_clicked(self, event):
		x, y = event.x, event.y
		figureid=int(3*y/unit_height)*3 + int(3*x/unit_width)
		scipy.misc.imsave('content.jpg', self.textImages[figureid])
		self.photoimagetext=ImageTk.PhotoImage(Image.fromarray(self.textImages[figureid]).resize((finalsize,finalsize)))
		self.selectedtext.config(image=self.photoimagetext)

	def scale_clicked(self, event):
		x, y = event.x, event.y
		figureid=int(3*y/unit_height)*3 + int(3*x/unit_width)
		scipy.misc.imsave('style.jpg', self.scaleImages[figureid])
		self.photoimagestyle=ImageTk.PhotoImage(Image.fromarray(self.scaleImages[figureid]).resize((finalsize,finalsize)))
		self.selectedstyle.config(image=self.photoimagestyle)

if __name__=='__main__':
	root = Tk()
	my_gui = GUI(root)

	root.mainloop()
