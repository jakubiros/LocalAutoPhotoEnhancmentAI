import os
import rawpy
import numpy as np
import cv2

class ImageInfo:
    def __init__(self, path):
        self.path = path
        self.img_name=self.__get_image_name()
        self.img_format = self.__get_image_format()

    def __get_image_name(self):
        if '/' in self.path:
            return self.path.split('/')[-1]
        elif '\\' in self.path:
            return self.path.split('\\')[-1]
        else:
            raise Exception('Invalid path')

    def __get_image_format(self):
        ext=self.path.lower()
        if ext.endswith(('.jpg', '.jpeg')):
            return '.jpg'
        elif ext.endswith(('.png', '.PNG')):
            return '.png'
        elif ext.endswith(('.nef', '.NEF')):
            return '.nef'
        elif ext.endswith(('.dng', '.DNG')):
            return '.dng'
        elif ext.endswith(('.tif', 'TIFF')):
            return '.tif'
        else:
            return None


class ImageLoader:
    def __init__(self, info: ImageInfo):
        self.info = info
        self.image=self.load_image()
        self.img_dtype=None

    def load_image(self):
        if self.info.img_format=='.nef' or self.info.img_format=='.dng':
            return self.__load_raw_image()
        else:
            return self.__load_nonraw_image()

    def __load_raw_image(self):
        img=rawpy.imread(self.info.path)
        img=img.postprocess(use_camera_wb=True, no_auto_bright=True, highlight_mode=rawpy.HighlightMode.Blend, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(1),
                           output_bps=16,output_color=rawpy.ColorSpace.Adobe, chromatic_aberration=(1,1))
        img=np.array(img, np.uint16)
        self.img_dtype=img.dtype
        return img

    def __load_nonraw_image(self):
        img=cv2.imread(self.info.path, cv2.IMREAD_UNCHANGED)
        #print(img.dtype)
        #self.img_dtype=img.dtype
        return img
