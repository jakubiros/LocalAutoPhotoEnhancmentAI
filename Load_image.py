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
            return self.path.split('/')[-1].split('.')[0]
        elif '\\' in self.path:
            return self.path.split('\\')[-1].split('.')[0]
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
        self.img_dtype = None
        self.image=self.load_image()

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
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_dtype=img.dtype
        return img

    def __load_nonraw_image(self):
        img=cv2.imread(self.info.path, cv2.IMREAD_UNCHANGED)
        self.img_dtype=img.dtype
        return img

class Normalizer:
    def __init__(self, img, image_loader: ImageLoader):
        self.image_loader=image_loader
        self.img=img
        #self.scaled_imaged=

    def scaling_01(self):
        if self.image_loader.img_dtype==np.uint16:
            return (self.img / 65535.0).astype(np.float32)
        else:
            return (self.img /255).astype(np.float32)

    def scaling_11(self):
        if self.image_loader.img_dtype==np.uint16:
            return ((self.img / 65535.0) * 2.0 - 1.0).astype(np.float32)
        else:
            return (self.img / 127.5 - 1).astype(np.float32)

    def back_scaler(self):
        if self.image_loader.img_dtype==np.uint16:
            if self.img.min() >= 0:
                return (self.img * 65535.0).clip(0, 65535).astype(np.uint16)
            else:
                return (((self.img + 1.0) / 2.0) * 65535.0).clip(0, 65535).astype(np.uint16)
        elif self.image_loader.img_dtype==np.uint8:
            if self.img.min() >= 0:
                return (self.img * 255).clip(0, 255).astype(np.uint8)
            else:
                return ((self.img + 1.0 ) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            raise Exception('Invalid image type')

class ConvertRGB2YUV:
    def __init__(self, original_img):
        self.original_img=original_img
        self.converted_img_yuv=self.__rgb_to_yuv(original_img)

    def __rgb_to_yuv(self,img):
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        self.converted_img_yuv = yuv_img[..., 0:1]
        return self.converted_img_yuv

class ConvertYUV2RGB:
    def __init__(self, original_img_rgb, y_image):
        self.original_img=original_img_rgb
        self.y_image=y_image
        self.converted_img_rgb=self.__yuv_to_rgb(original_img_rgb)

    def __yuv_to_rgb(self, img):
        original_yuv=cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        original_yuv[..., 0:1]=self.y_image
        self.converted_img_rgb=cv2.cvtColor(original_yuv, cv2.COLOR_YUV2RGB)
        return np.clip(self.converted_img_rgb, 0.0, 1.0)

class SaveImage:
    def __init__(self, info: ImageInfo, path):
        self.info=info
        self.path=path

    def save_image(self, img):
        print(self.path + self.info.img_name +'_AIenhanced'+'.jpg')
        if self.info.img_format=='.jpg':
            cv2.imwrite(self.path+'/' + self.info.img_name +'_AIenhanced'+'.jpg', img)
        else:
            cv2.imwrite(self.path +'/' + self.info.img_name + '_AIenhanced' + '.png', img)