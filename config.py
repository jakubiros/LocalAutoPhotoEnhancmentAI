import os
import platform
import sys

class SystemInfo:
    def __init__(self):
        self.system=platform.system()
        self.RAM_gb=self.__get_ram_info()
        self.device=self.__get_device_info()

    def __get_ram_info(self):
        try:
            ram_bytes=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            ram_gb=round(ram_bytes/ (1024**3), 1)
        except (ValueError, AttributeError):
            ram_gb=0
        return ram_gb

    def __get_device_info(self):
        if self.system=='Darwin':
            return 'mac'
        elif self.system=='Linux':
            return 'rpi5'
        else:
            return 'generic_linux'

    def __call__(self, *args, **kwargs):
        return {'system':self.system, 'RAM_gb':self.RAM_gb, 'device':self.device}

