#
# Simple Python program to calculate elements in the Mandelbrot set.
#
import numpy as np
from numba import cuda, int32
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
        Given the real and imaginary parts of a complex number,
        determine if it is a candidate for membership in the
        Mandelbrot set given a fixed number of iterations.
        '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit()
def compute_mandel(min_x, max_x, min_y, max_y, image_global, iters):
    '''
        Calculate the mandel value for each element in the
        image array. The real and imag variables contain a
        value for each element of the complex space defined
        by the X and Y boundaries (min_x, max_x) and
        (min_y, max_y).
        
        Each thread is responsible for a small part of the image
        which locates at the positon of the thread in the whole grid.
        Therefore, the starting coordinates can be obtained as (cuda.grid(2))
        and the size of the small block is calculated as
        (image.shape[0]/(grid_w*block_w)), image.shape[1]/(grid_h*block_h)).
        '''
    
    block_w, block_h = cuda.blockDim.x, cuda.blockDim.y
    grid_w, grid_h = cuda.gridDim.x, cuda.gridDim.y
    row, col = cuda.grid(2)
    
    # create shared memory for image #
    image = cuda.shared.array(shape=image_global.shape, dtype = int32)
    
    increment_h = int(image.shape[0]/(grid_w*block_w))
    start_h = increment_h * row
    end_h = start_h + increment_h
    increment_w = int(image.shape[1]/(grid_h*block_h))
    start_w = increment_w * col
    end_w = start_w + increment_w
    
    pixel_size_x = (max_x - min_x) / image.shape[1]
    pixel_size_y = (max_y - min_y) / image.shape[0]
    
    for x in range(start_w, end_w):
        real = min_x + x * pixel_size_x
        for y in range(start_h, end_h):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)

    # wait for all threads in the block to finish #
    cuda.syncthreads()
    
    # the first thread of this block writes the result into global variable #
    if row%block_w==0 and col%block_h == 0:
        for i in range(increment_h*row, increment_h*(row+block_w)):
            for j in range(increment_w*col, increment_w*(col+block_h)):
                image_global[i][j] = image[i][j]


if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    
    blockdim = (32, 8)
    griddim = (32, 16)
    
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20)
    image_global_mem.copy_to_host()
    imshow(image)
    show()
