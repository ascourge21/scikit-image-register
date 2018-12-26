import math

import numpy as np
import skimage.util
import skimage.transform


# TODO (nripesh): add tests     
def resize_image(im, new_size):
    # because rescale alters size, crop or pad to keep it consistent
    # assumes image is of dim 3
    orig_row, orig_col = im.shape[0], im.shape[1]
    assert orig_row > 0
    assert orig_col > 0
    new_row, new_col = new_size
    
    if new_row > orig_row:  # assuming rows/cols scale similarly
        row_pad_before = math.ceil((new_row - orig_row) / 2)
        row_pad_after = new_row - orig_row - row_pad_before
        
        col_pad_before = math.ceil((new_col - orig_col) / 2)
        col_pad_after = new_col - orig_col - col_pad_before
        
        if len(im.shape) == 3:
            im_resized = skimage.util.pad(im,
                ((row_pad_before, row_pad_after),
                 (col_pad_before, col_pad_after),
                 (0, 0)), mode='constant',
            )
        else:
            im_resized = skimage.util.pad(im,
                ((row_pad_before, row_pad_after),
                 (col_pad_before, col_pad_after)),
                mode='constant',
            )           
    else:
        row_crop_before = math.ceil((orig_row - new_row) / 2)
        row_crop_after = orig_row - new_row -  row_crop_before
        
        col_crop_before = math.ceil((orig_col - new_col) / 2)
        col_crop_after = orig_col - new_col - col_crop_before
        
        if len(im.shape) == 3:
            im_resized = skimage.util.crop(im,
                ((row_crop_before, row_crop_after),
                 (col_crop_before, col_crop_after),
                 (0, 0))
            )
        else:
            im_resized = skimage.util.crop(im,
                ((row_crop_before, row_crop_after),
                 (col_crop_before, col_crop_after)),
            )
    assert im_resized.shape[0] == new_size[0]
    assert im_resized.shape[1] == new_size[1]
    return im_resized


# TODO (nripesh): add tests 
# TODO : separate the image difference to different function
#        with other differences also avaialble.
# refactor to use the same transformation as in affine.
def ssd_scale(scale_factor, im_fixed, im_moving, im_pad_size,
           xx=None, yy=None, gradient=False):
    # re-initialize if problematic
    if scale_factor < 0:
        scale_factor = [np.abs(np.random.randn())]
        print(scale_factor)
        
    im_moving_regis = skimage.transform.rescale(
        im_moving, scale=scale_factor, multichannel=True,
        mode='constant', anti_aliasing=True
    )
    
    im_movig_regis = resize_image(im_moving_regis, im_pad_size)
    row, col, channel = im_movig_regis.shape

    q1 = im_fixed - im_movig_regis
    ssd = np.sum(np.square(q1)) / (row * col * channel)
    
    if xx is None:
        xx, yy = np.meshgrid(
            np.arange(im_pad_size[0]), np.arange(im_pad_size[1]))

    if not gradient:
        print(scale_factor, ssd)
        return ssd

    grad = 0
    for c in range(channel):
        # grad along single dimension should be enough
        grad = np.gradient(im_movig_regis[:, :, c])
        grad_x, grad_y = grad[0], grad[1]

        # this is specific to scaling
        grad_x = np.multiply(grad_x / scale_factor[0], xx)
        grad_y = np.multiply(grad_y / scale_factor[0], yy)

        grad += ( 
            - 2 * np.sum(np.multiply(
                q1[:, :, c], grad_x + grad_y)) / (row * col * channel))
    
    print(scale_factor, ssd, grad)
    return ssd, grad