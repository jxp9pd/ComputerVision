import numpy, skimage, skimage.io, pylab, scipy.ndimage.filters, math
numpy.set_printoptions(threshold=numpy.nan)

#A is the original image
A=skimage.io.imread('/Users/johnp_000/Dropbox/UVA 2016-2017/CS 4501/pics/audi.jpg')
A=skimage.img_as_float(A)
pylab.imshow(A)
pylab.show()
WIDTH = len(A[0])
HEIGHT = len(A)
print (WIDTH)
#averages three color channels according to the human eye
def grayscale(image):
	return (0.21*image[0] + 0.72*image[1] + 0.07*image[2]) 
grey = numpy.zeros((HEIGHT, WIDTH)) 
for rownum in range(HEIGHT):
   for colnum in range(WIDTH):
      grey[rownum][colnum] = grayscale(A[rownum][colnum])
#Convolves 2D luminance w/ Gaussian
gaussianK = [[0.0625, 0.125, 0.0625],
	  		 [0.125, 0.25, 0.125],
	  		 [0.0625, 0.125, 0.0625]]
blur = scipy.ndimage.filters.convolve(grey, gaussianK)

sobelx = [[-1, 0, 1],
		  [-2, 0, 2],
		  [-1, 0, 1]]

sobely = [[ 1, 2, 1],
		  [ 0, 0, 0],
		  [-1,-2,-1]]

xgradient = scipy.ndimage.filters.convolve(blur, sobelx)
ygradient = scipy.ndimage.filters.convolve(blur, sobely)
#Calculate magnitude and angles
angle = numpy.zeros((HEIGHT,WIDTH))
angle = numpy.arctan(ygradient/xgradient)

xSquared = numpy.square(xgradient)
ySquared = numpy.square(ygradient)
summed = xSquared+ySquared
magnitude = numpy.sqrt(summed)
magnitude= skimage.img_as_float(magnitude)
pylab.imshow(magnitude)
pylab.show()

class Point:
	row = 0
	col = 0
	mag=0
	visited = False;
	def __init__(self,row,col,mag):
		self.row = row
		self.col = col
		self.mag = mag
	def getMag(self):
		return self.mag
	def setVisited(self):
		visited=True

def getNeighbors(center, magnitude):
	result = []
	row = center.row
	col = center.col
	if(col+1<WIDTH):
		result.append(Point(row,col+1,magnitude[row][col+1].getMag()))
	if(col-1>0):
		result.append(Point(row,col-1,magnitude[row][col-1].getMag()))
	if(row+1<HEIGHT):
		result.append(Point(row,col+1,magnitude[row+1][col].getMag()))
	if(row-1>0):
		result.append(Point(row-1,col,magnitude[row-1][col].getMag()))
	if(row+1<HEIGHT and col+1<WIDTH):
		result.append(Point(row+1,col+1,magnitude[row+1][col+1].getMag()))
	if(row>0 and col+1<WIDTH):
		result.append(Point(row-1,col+1,magnitude[row-1][col+1].getMag()))
	if(row+1<HEIGHT and col+1<WIDTH):
		result.append(Point(row+1,col-1,magnitude[row+1][col-1].getMag()))
	if(row-1>0 and col-1>0):
		result.append(Point(row-1,col+1,magnitude[row-1][col-1].getMag()))
	return result

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
	elif(num+1>quadrants):
		return 1
	else:
		return num+1

qAngles = numpy.zeros((HEIGHT, WIDTH))
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		qAngles[rownum][colnum] = quantize(angle[rownum][colnum],8)
#thinning
qCopy = numpy.copy(magnitude)

#terribly space inefficient
def thin(pixel, direction, x, y):
	row = x
	col = y
	row2 = x
	col2 = y
	if (direction==1 or direction==5):
		col-=1
		col2+=1
	elif (direction==3 or direction==7):
		row-=1
		row2+=1
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
	if (row>0 and row<HEIGHT and col>0 and col<WIDTH):
		neighbor1 = magnitude[row][col]
	if (row2>0 and row2<HEIGHT and col2>0 and col2<WIDTH):
		neighbor2 = magnitude[row2][col2]
	if (magnitude[x][y]<neighbor1 or magnitude[x][y]<neighbor2):
		qCopy[x][y]=0
	# elif (max(neighbor2,neighbor1,magnitude[x][y])==neighbor2):
	# 	qCopy[x][y]=0

for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		thin(magnitude[rownum][colnum],qAngles[rownum][colnum], rownum, colnum)

thinnedImage = skimage.img_as_float(qCopy)
pylab.imshow(thinnedImage)
pylab.show()


#Thresholding
THIGH = 0.3
TLOW = 0.15
def dfs(threshold,strongEdge):
	row = strongEdge.row
	col = strongEdge.col
	threshold[row][col].setVisited()
	neighbors = getNeighbors(strongEdge, threshold)
	for i in range(0, len(neighbors)):
		if(neighbors[i].visited != True):
			border = neighbors[i]
			if(border.mag>TLOW and border.mag <THIGH):
				threshold[border.row][border.col].mag = 1
				dfs(threshold,border)

def stackDfs(threshold, strongEdge):
	stack = [strongEdge]
	while(len(stack)>0):
		currEdge = stack.pop()
		if(currEdge.visited!=True):
			currEdge.setVisited()
			for edge in (getNeighbors(currEdge, threshold)):
				if(edge.getMag()<THIGH and edge.getMag()>TLOW
					and edge.row<HEIGHT and edge.col<WIDTH
					and edge.row>0 and edge.col>0):
					threshold[edge.row][edge.col].mag=1
					stack.append(edge)


threshold = [[Point(0,0,0) for j in range(WIDTH)] for i in range(HEIGHT)]
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		pixel = Point(rownum, colnum, qCopy[rownum][colnum])
		if (pixel.getMag()<TLOW):
			pixel.mag = 0
		elif (pixel.getMag()>THIGH):
			pixel.mag = 1
		threshold[rownum][colnum] = pixel

for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		if (threshold[rownum][colnum].getMag() > THIGH):
			stackDfs(threshold,threshold[rownum][colnum])

cannyEdge = numpy.zeros((HEIGHT, WIDTH))
for rownum in range(HEIGHT):
	for colnum in range(WIDTH):
		if(threshold[rownum][colnum].getMag()!=1):
			threshold[rownum][colnum].mag = 0
		cannyEdge[rownum][colnum] = threshold[rownum][colnum].getMag()

cannyEdge = skimage.img_as_float(cannyEdge)
pylab.imshow(cannyEdge)
pylab.show()
