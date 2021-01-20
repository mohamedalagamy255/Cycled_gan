#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import skimage.io as iio
import cv2

from imlib import dtype


def imread(path, as_gray=False, **kwargs):
    """Return a float64 image in [-1.0, 1.0]."""
    image = cv2.imread(path,**kwargs)
    if as_gray:
        image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    else :
        pass
    
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    elif image.dtype == np.uint16:
        image = image / 32767.5 - 1
    elif image.dtype in [np.float32, np.float64]:
        image = image * 2 - 1.0
    else:
        raise Exception("Inavailable image dtype: %s!" % image.dtype)
    return image


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    #cv2.imsave(path, dtype.im2uint(image), quality=quality, **plugin_args)
    image = dtype.im2uint(image)
    cv2.imwrite(path, image)


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    cv2.imshow("image" , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #iio.imshow(dtype.im2uint(image))


#show = iio.show

