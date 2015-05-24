from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

def matting(b1, b2, c1, c2):
    '''Computes the triangulation matting equation
    
       Return:
       fg: foreground image
       alpha: alpha image '''
    b1_r, b1_g, b1_b = b1[:,:,0], b1[:,:,1], b1[:,:,2]    
    b2_r, b2_g, b2_b = b2[:,:,0], b2[:,:,1], b2[:,:,2]    
    c1_r, c1_g, c1_b = c1[:,:,0], c1[:,:,1], c1[:,:,2]
    c2_r, c2_g, c2_b = c2[:,:,0], c2[:,:,1], c2[:,:,2]
    
    img_shape = b1.shape # all images have same shape
    fg = np.zeros(img_shape)
    alpha = np.zeros(img_shape[:2])
    
    matrix = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
    
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            a = np.array([[b1_r[i,j]],
                          [b1_g[i,j]],
                          [b1_b[i,j]],
                          [b2_r[i,j]],
                          [b2_g[i,j]],
                          [b2_b[i,j]]])
            b = np.array([[c1_r[i,j]-b1_r[i,j]],
                          [c1_g[i,j]-b1_g[i,j]],
                          [c1_b[i,j]-b1_b[i,j]],
                          [c2_r[i,j]-b2_r[i,j]],
                          [c2_g[i,j]-b2_g[i,j]],
                          [c2_b[i,j]-b2_b[i,j]]])
            A = np.hstack((matrix, -1*a))
            x = np.clip(np.dot(np.linalg.pinv(A),b), 0.0, 1.0)
            fg[i,j] = np.array([x[0][0], x[1][0], x[2][0]])
            alpha[i,j] = x[3]
    return fg, alpha
    
def multiply_alpha(alpha, b):
    '''Multiplies (1-alpha) and the background image
    
       Returns
       c: (1-alpha) * background'''
    img_shape = b.shape
    c = np.zeros(img_shape)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            # alpha image has one value for each pixel
            # unlike the background image which has three values - r,g,b
            c[i][j] = b[i][j]*(1.0-alpha[i][j])
    return c

if __name__ == '__main__':
    window = np.array(Image.open('window.jpg'))/255.0
    
    b1 = np.array(Image.open('flowers-backA.jpg'))/255.0
    b2 = np.array(Image.open('flowers-backB.jpg'))/255.0
    c1 = np.array(Image.open('flowers-compA.jpg'))/255.0
    c2 = np.array(Image.open('flowers-compB.jpg'))/255.0
    fg, alpha = matting(b1,b2,c1,c2)
    imsave('flowers-alpha.jpg', alpha, cmap=cm.gray)
    imsave('flowers-foreground.jpg', fg)
    b = multiply_alpha(alpha, window)
    composite = fg + b
    plt.show(imshow(composite))
    imsave('flowers-composite.jpg', composite)
    
    b1 = np.array(Image.open('leaves-backA.jpg'))/255.0
    b2 = np.array(Image.open('leaves-backB.jpg'))/255.0
    c1 = np.array(Image.open('leaves-compA.jpg'))/255.0
    c2 = np.array(Image.open('leaves-compB.jpg'))/255.0
    fg, alpha = matting(b1,b2,c1,c2)
    imsave('leaves-alpha.jpg', alpha, cmap=cm.gray)
    imsave('leaves-foreground.jpg', fg)
    b = multiply_alpha(alpha, window)
    composite = fg + b
    plt.show(imshow(composite))
    imsave('leaves-composite.jpg', composite)
