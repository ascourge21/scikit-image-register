import scipy.optimize
import skimage.transform
import utils


# constants
# Make this variable -> change
TO_PAD_SIZE = (1200, 1200)


def im_register_scale(im_fixed, im_moving,
	pad_size=TO_PAD_SIZE, options={}):
	
	im_f_shape = im_fixed.shape
	im_m_shape = im_moving.shape
	assert im_f_shape == im_m_shape

	# random initialization and assignment
	scale_init = 1.25
	error_function = lambda x: utils.ssd_scale(
		x, im_fixed, im_moving, pad_size)
	scale_optimized = scipy.optimize.minimize(
	    error_function, scale_init, method='Nelder-Mead',
	    options=options, jac=False)

	# apply solved registration
	im_registered = skimage.transform.rescale(
	    im_moving, scale_optimized.x, multichannel=True,
	    mode='constant', anti_aliasing=True
	)
	
	return utils.resize_image(im_registered, pad_size)
