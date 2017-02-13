import numpy, skimage, skimage.io, pylab, scipy.ndimage.filters

#A is the original image
A=skimage.io.imread('/Users/johnp_000/Dropbox/UVA 2016-2017/CS 4501/lenna.png')
A=skimage.img_as_float(A)

#averages three color channels according to the human eye
def grayscale(image):
	return (0.21*image[0] + 0.72*image[1] + 0.07*image[2])

C=A 
grey = numpy.zeros((len(C), len(C[1]))) 
for rownum in range(len(C)):
   for colnum in range(len(C[rownum])):
      grey[rownum][colnum] = grayscale(C[rownum][colnum])
grey=skimage.img_as_float(grey)
# pylab.imshow(grey)
# pylab.show()

#Convolves 2D luminance w/ Gaussian
K = numpy.ones((20,20)) / 20**2
gaussianK = [[0.0625, 0.125, 0.0625],
	  [0.125, 0.25, 0.125],
	  [0.0625, 0.125, 0.0625]]
blur = scipy.ndimage.filters.convolve(grey, gaussianK)
#blur=skimage.img_as_float(blur)
# pylab.imshow(grad)
# pylab.show()

sobelx = [[-1, 0, 1],
		  [-2, 0, 1],
		  [-1, 0, -1]]
sobely = [[1, 2, 1],
		  [0, 0, 0],
		  [-1, -2, -1]]		 

xgradient = scipy.ndimage.filters.convolve(blur, sobelx)
ygradient = scipy.ndimage.filters.convolve(blur, sobely)
# xgradient = skimage.img_as_float(xgradient)
# ygradient = skimage.img_as_float(ygradient)
# pylab.imshow(xgradient)
# pylab.show()
# pylab.imshow(ygradient)
# pylab.show()

#Calculate magnitude and angles
angle = numpy.arctan2(ygradient, xgradient)
image = skimage.img_as_float(angle)
# pylab.imshow(image)
# pylab.show()

xSquared = numpy.square(xgradient)
ySquared = numpy.square(ygradient)
summed = xSquared+ySquared
magnitude = numpy.sqrt(summed)
#magnitude = skimage.img_as_float(magnitude)
if (numpy.array_equal(magnitude, grey)):
	print("Fuck me")
else:
	print("There's at least a small difference")

#Nonmaximum Supression
def quantize(angle, quadrants):
	section = -1 * numpy.pi
	num = 1
	addend = numpy.pi/(quadrants/2)
	while (angle>section+addend and section<numpy.pi):
		section += addend
		num+=1
	if(angle-section < angle-section-addend):
		return num
	else:
		return num+1
qAngles = numpy.zeros((len(image), len(image[0])))
for rownum in range(len(image)):
	for colnum in range(len(image)):
		qAngles[rownum][colnum] = quantize(angle[rownum][colnum],8)
#thinning
qCopy = magnitude
#terribly space inefficient
#if answers are wrong check here
def thin(pixel, direction, x, y, magnitude):
	row = x
	col = y
	row2 = x
	col2 = y
	#horizontal gradient
	if (direction==1 or direction==5):
		row-=1
		row2+=1
	elif (direction==3 or direction==7):
		col-=1
		col2+=1
	elif (direction==2 or direction==6):
		row-=1
		col-=1
		row2+=1
		col2+=1
	elif (direction==4 or direction==8):
		row-=1
		col+=1
		row2+=1
		col2-=1
	neighbor1=0
	neighbor2=0
	check1 = False
	check2 = False
	if (row>0 and row<len(magnitude) and col>0 and col<len(magnitude[0])):
		neighbor1 = magnitude[row][col]
		check1 = True
	if (row2>0 and row2<len(magnitude) and col2>0 and col2<len(magnitude[0])):
		neighbor2 = magnitude[row2][col2]
		check2 = True
	if (max(neighbor2,neighbor1,magnitude[x][y])==neighbor1):
		if (check2):
			qCopy[row2][col2]=0
		qCopy[x][y]=0
	elif (max(neighbor2,neighbor1,magnitude[x][y])==neighbor2):
		if (check1):
			qCopy[row][col]=0
		qCopy[x][y]=0
	else:
		if (check1):
			qCopy[row][col]=0
		if (check2):
			qCopy[row2][col2]=0

for rownum in range(len(image)):
	for colnum in range(len(image)):
		thin(magnitude[rownum][colnum],qAngles[rownum][colnum], rownum, colnum, magnitude)

thinnedImage = skimage.img_as_float(qCopy)
pylab.imshow(thinnedImage)
pylab.show()






