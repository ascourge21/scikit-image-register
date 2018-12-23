import math


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import skimage.data
import skimage.transform
import utils

# image examples
im = skimage.data.astronaut()

# constants
TO_PAD_SIZE = (1200, 1200)
scale_true = .5
ORIG_IM_SIZE = (im.shape[0], im.shape[1])
xx, yy = np.meshgrid(
    np.arange(TO_PAD_SIZE[0]), np.arange(TO_PAD_SIZE[1]))

# assign target and moving
im_fixed = skimage.transform.rescale(
    im, scale=scale_true, clip=True, multichannel=True,
    mode='constant', anti_aliasing=True
)
im_fixed = utils.resize_image(im_fixed, TO_PAD_SIZE)
im_moving = utils.resize_image(im, TO_PAD_SIZE) / 255

# random initialization and assignment
scale_init = 1.25
options = {'maxiter': 30}
error_function = lambda x: utils.ssd_scale(
	x, im_fixed, im_moving, TO_PAD_SIZE, xx, yy)
scale_optimized = scipy.optimize.minimize(
    error_function, scale_init, method='Nelder-Mead',
    options=options, jac=False)
print(scale_optimized)

# apply solved registration
im_registered = skimage.transform.rescale(
    im_moving, scale_optimized.x, multichannel=True,
    mode='constant', anti_aliasing=True
)
im_registered = utils.resize_image(im_registered, TO_PAD_SIZE)

plt.figure(1)
plt.title('target (transformed)')
plt.imshow(im_fixed)
plt.figure(2)
plt.title('moving (orig)')
plt.imshow(im_moving)
plt.figure(3)
plt.title('registered')
plt.imshow(im_registered)
plt.show()