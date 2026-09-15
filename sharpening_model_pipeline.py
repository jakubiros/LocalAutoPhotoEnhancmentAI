import numpy as np
from load_model import KerasModelLoader, LiteRTModelLoader
from utils import Utils, SimpleGate, LayerScale

class SharpenImageStandard:
    def __init__(self, image, patch_size=512, min_overlap=96, batch_size=16):
        self.image = image
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.batch_size = batch_size
        self.sharpened_image = None

    def __call__(self):
        model_loader=KerasModelLoader('/Users/jakubbiros/Documents/jupyter_notebooks/AutoImageProcessing/models/NAFNet_L-12-patch-denoise-v2.keras',
                                      {SimpleGate:'SimpleGate', LayerScale:'LayerScale'})
        model = model_loader.load()

        y_coords = Utils.get_grid_coordinates(self.image.shape[0], self.patch_size, min_overlap=self.min_overlap)
        x_coords = Utils.get_grid_coordinates(self.image.shape[1], self.patch_size, min_overlap=self.min_overlap)

        print(f"Siatka: {len(y_coords)} x {len(x_coords)} = {len(y_coords) * len(x_coords)} kafelków")

        reconstructed = np.zeros_like(self.image, dtype=np.float32)
        weight_map = np.zeros_like(self.image, dtype=np.float32)
        hann_window = Utils.create_hann_window(self.patch_size)

        patches = []
        coords = []

        for y in y_coords:
            for x in x_coords:
                patches.append(self.image[y:y + self.patch_size, x:x + self.patch_size, :])
                coords.append((y, x))

                if len(patches) == self.batch_size:
                    predictions = model.predict_on_batch(np.stack(patches))

                    for pred, (py, px) in zip(predictions, coords):
                        reconstructed[py:py + self.patch_size, px:px + self.patch_size, :] += pred * hann_window
                        weight_map[py:py + self.patch_size, px:px + self.patch_size, :] += hann_window

                    patches.clear()
                    coords.clear()

        if patches:
            predictions = model.predict_on_batch(np.stack(patches))

            for pred, (py, px) in zip(predictions, coords):
                reconstructed[py:py + self.patch_size, px:px + self.patch_size, :] += pred * hann_window
                weight_map[py:py + self.patch_size, px:px + self.patch_size, :] += hann_window

        weight_map[weight_map == 0] = 1.0
        self.sharpened_image = reconstructed / weight_map
        model_loader.unload()

        return self.sharpened_image

class SharpenImageRPi:
    def __init__(self, image, patch_size=512, min_overlap=64):
        self.image = image
        self.patch_size = patch_size
        self.min_overlap = min_overlap
        self.sharpened_image = None

    def __call__(self):
        model=LiteRTModelLoader('', num_threads=4)
        interpreter = model.load()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']


        y_coords = Utils.get_grid_coordinates(self.image.shape[0], self.patch_size, min_overlap=64)
        x_coords = Utils.get_grid_coordinates(self.image.shape[1], self.patch_size, min_overlap=64)
        total_patches = len(y_coords) * len(x_coords)
        print(
            f'Siatka: {len(y_coords)} w pionie x {len(x_coords)} w poziomie ='
            f' {total_patches} kafelków'
        )

        reconstructed = np.zeros_like(self.image, dtype=np.float32)
        weight_map = np.zeros_like(self.image, dtype=np.float32)
        hann_window = Utils.create_hann_window(self.patch_size)

        input_buffer = np.empty(
            (1, self.patch_size, self.patch_size, 1), dtype=input_details[0]['dtype']
        )

        patch_idx = 0
        for y in y_coords:
            for x in x_coords:
                patch_idx += 1
                patch = self.image[y: y + self.patch_size, x: x + self.patch_size, :]
                input_buffer[0] = patch

                interpreter.set_tensor(input_index, input_buffer)
                interpreter.invoke()
                pred_patch = interpreter.get_tensor(output_index)[0]

                reconstructed[y: y + self.patch_size, x: x + self.patch_size, :] += (
                        pred_patch * hann_window
                )
                weight_map[y: y + self.patch_size, x: x + self.patch_size, :] += hann_window

        weight_map[weight_map == 0] = 1.0
        self.sharpened_image = reconstructed / weight_map
        interpreter.unload()

        return self.sharpened_image
