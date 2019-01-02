from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import time


def matting(b1, b2, c1, c2):
    '''Computes the triangulation matting equation
       
       Param:
       b1: background image 1
       b2: background image 2
       c1: composite image 1 (back + front)
       c2: composite image 2 (back + front)

       Returns:
       fg: foreground image
       alpha: alpha image '''
    print("[*] Start matting...")
    time_start = time.time()
    img_shape = b1.shape  # all images have same shape
    H, W, C = img_shape
    assert C == 3, "ONLY support RGB format images!"
    B = np.concatenate([b1, b2], axis=2).reshape(H, W, 2*C, 1)  # stack along the channel axis
    I = np.concatenate([c1, c2], axis=2).reshape(H, W, 2*C, 1)
    R = I - B
    A = np.zeros((H, W, 2*C, 3))
    m = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
    A = A + m
    A = np.concatenate([A, -B], axis=3)
    
    x_pinv = np.linalg.pinv(A)
    x_dot = np.matmul(x_pinv, R)
    X = np.clip(x_dot, 0.0, 1.0).squeeze(axis=3)
    fg = X[:, :, :3]
    alpha = X[:, :, 3]
    
    print("[*] Matting completed with time elapse: {} sec".format(time.time() - time_start))
    return fg, alpha


if __name__ == '__main__':
    window = np.array(Image.open('window.jpg'))/255.0
    
    b1 = np.array(Image.open('flowers-backA.jpg'))/255.0
    b2 = np.array(Image.open('flowers-backB.jpg'))/255.0
    c1 = np.array(Image.open('flowers-compA.jpg'))/255.0
    c2 = np.array(Image.open('flowers-compB.jpg'))/255.0
    fg, alpha = matting(b1,b2,c1,c2)
    imsave('flowers-alpha.jpg', alpha, cmap=cm.gray)
    imsave('flowers-foreground.jpg', fg)
    b = (1.0 - alpha).reshape(alpha.shape[0], alpha.shape[1], 1) * window
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
    b = (1.0 - alpha).reshape(alpha.shape[0], alpha.shape[1], 1) * window
    composite = fg + b
    plt.show(imshow(composite))
    imsave('leaves-composite.jpg', composite)
