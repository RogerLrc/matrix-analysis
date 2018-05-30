import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def svd(image,num):
    u,sigma,w = np.linalg.svd(image)
    u1 = u[:,:num]
    sigma1 = np.diag(sigma[:num],0)
    w1 = w[:num,:]
    new_image = u1.dot(sigma1).dot(w1)
    new_image[new_image < 0] = 0
    new_image[new_image > 255] = 255

    return np.rint(new_image).astype('uint8')

if __name__ == '__main__':
    image = Image.open('1.png')
    image_mat = np.array(image)
    [a,b,c] = [image_mat[:,:,0],image_mat[:,:,1],image_mat[:,:,2]]
    
    for i in range(6):
        number = 5*i
        new_image_mt = np.stack([svd(a,number),svd(b,number),svd(c,number)],2)
        new_image = Image.fromarray(new_image_mt)
        new_image.save('svd_'+str(number)+'.png')
   # plt.imshow(new_image)

