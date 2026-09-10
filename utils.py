import numpy as np


class Utils:

    @staticmethod
    def create_hann_window(size):
        w = np.hanning(size)
        window_2d = np.outer(w, w)
        return np.expand_dims(window_2d, axis=-1).astype(np.float32)

    @staticmethod
    def get_grid_coordinates(image_dim, patch_size, min_overlap=64):
        if image_dim <= patch_size:
            return [0]

        n_patches = int(
            np.ceil((image_dim - min_overlap) / (patch_size - min_overlap))
        )
        step = (image_dim - patch_size) / (n_patches - 1)
        return [int(round(i * step)) for i in range(n_patches)]