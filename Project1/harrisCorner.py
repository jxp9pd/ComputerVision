import numpy, skimage, skimage.io, pylab, scipy.ndimage.filters, math, time
start_time = time.time()
numpy.set_printoptions(threshold=numpy.nan)
#A is the original image
A=skimage.io.imread('/Users/johnp_000/Dropbox/UVA 2016-2017/CS 4501/pics/audi.jpg')
A=skimage.img_as_float(A)
# pylab.imshow(A)
# pylab.show()
HEIGHT = len(A)
WIDTH = len(A[0])
CORNERT = 0.45
if (WIDTH==1024):
	CORNERT = 0.1
#averages three color channels according to the human eye
def grayscale(image):
	return (0.21*image[0] + 0.72*image[1] + 0.07*image[2])
C=A 
grey = numpy.zeros((HEIGHT, WIDTH)) 
for rownum in range(HEIGHT):
   for colnum in range(WIDTH):
      grey[rownum][colnum] = grayscale(C[rownum][colnum])
#grey=skimage.img_as_float(grey)
# pylab.imshow(grey)
# pylab.show()
#Convolves 2D luminance w/ Gaussian
gaussianK = [[0.0625, 0.125, 0.0625],
	  [0.125, 0.25, 0.125],
	  [0.0625, 0.125, 0.0625]]
blur = scipy.ndimage.filters.convolve(grey, gaussianK)

sobelx = [[-1, 0, 1],
		  [-2, 0, 2],
		  [-1, 0, 1]]

sobely = [[1, 2, 1],
		  [0, 0, 0],
		  [-1, -2, -1]]

xgradient = scipy.ndimage.filters.convolve(blur, sobelx)
ygradient = scipy.ndimage.filters.convolve(blur, sobely)

#Calculate magnitude and angles
angle = numpy.zeros((HEIGHT,WIDTH))
angle = numpy.arctan(ygradient/xgradient)
image = skimage.img_as_float(angle)

xSquared = numpy.square(xgradient)
ySquared = numpy.square(ygradient)
summed = xSquared+ySquared
magnitude = numpy.sqrt(summed)
# magnitude = skimage.img_as_float(magnitude)
# pylab.imshow(magnitude)
# pylab.show()

class Point:
	row = 0
	col = 0
	mag=0
	Fx=0
	Fy=0
	eigenval = 0
	visited = False;
	def __init__(self,row,col,Fx,Fy,eigenval):
		self.row = row
		self.col = col
		self.Fx = Fx
		self.Fy = Fy
		self.eigenval = eigenval
	def getMag(self):
		return self.mag
	def setVisited(self):
		visited=True

def spatialAverage(window, m):
	size = (2*m+1)**2
	avg = numpy.zeros((2,2))
	fx2 = 0
	fxy = 0
	fy2 = 0
	for i in range(0,len(window)):
		fx2 += (window[i].Fx)**2
		fxy += window[i].Fx * window[i].Fy
		fy2 += (window[i].Fy)**2
	avg[0][0] = fx2/size
	avg[0][1] = fxy/size
	avg[1][0] = fxy/size
	avg[1][1] = fy2/size
	return avg;

covImage = [[numpy.zeros((2,2)) for j in range(WIDTH)] for i in range(HEIGHT)]

def isNeighbor(original, possible):
	if (original.row - possible.row <=1 and original.col -possible.col <=1
		and original.row - possible.row >=-1 and original.col -possible.col >=-1):
		return True
	else:
		return False


def addTargets(corners, image):
	for i in range(0, len(corners)):
		x = corners[i]
		for u in range(x.row-1, x.row+1):
			for v in range(x.col-1, x.col+1):
				if (u<HEIGHT and v<WIDTH and u>0 and v>0):
					image[u,v,0] = 0
					image[u,v,1] = 0
					image[u,v,2] = 1
eigenMatrix = numpy.zeros((HEIGHT, WIDTH))

def covariance(m,xgradient,ygradient):
	corners = []
	sum =0
	for i in range(HEIGHT):
		for j in range(WIDTH):
			window = []
			for u in range(i-m, i+m+1):
				for v in range(j-m, j+m+1):
					if(u>0 and u<HEIGHT and v>0 and v<WIDTH):
						window.append(Point(u, v, xgradient[u][v], ygradient[u][v],0))
			covMatrix = spatialAverage(window, m)
			w,v = numpy.linalg.eig(covMatrix)
			eigenMatrix[i][j] = min(w)
			if(min(w)>CORNERT):
				corners.append(Point(i, j, xgradient[i][j], ygradient[i][j], min(w)))
	return corners
corners = covariance(4, xgradient, ygradient)
print(str(len(corners)) + " corners detected")

def searchList(list, coordinate):
	for i in list:
		if (coordinate == i):
			return True
	return False
sortedL = sorted(corners, key=lambda Point: Point.eigenval)
result = []
removed = []
for x in sortedL:
	if (searchList(removed, (x.row, x.col)) != True):
		result.append(x)
		removed.append((x.row+1, x.col))
		removed.append((x.row+1, x.col+1))
		removed.append((x.row+1, x.col-1))
		removed.append((x.row-1, x.col))
		removed.append((x.row-1, x.col+1))
		removed.append((x.row-1, x.col-1))
		removed.append((x.row, x.col+1))
		removed.append((x.row, x.col-1))

print(str(len(result)) + " corners detected after surpression")
image = addTargets(result, A)
print("--- %s seconds ---" % (time.time() - start_time))
pylab.imshow(A)
pylab.show()

