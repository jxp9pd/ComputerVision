import numpy
import skimage, skimage.io, pylab, scipy.ndimage.filters

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
grad = scipy.ndimage.filters.convolve(grey, K)
grad=skimage.img_as_float(grad)
grad = numpy.gradient(grad)

pylab.imshow(grad)
pylab.show()
