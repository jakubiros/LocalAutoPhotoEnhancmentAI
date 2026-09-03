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
        else:
            return None


class ImageLoader:
    def __init__(self, info: ImageInfo):

