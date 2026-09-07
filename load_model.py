from abc import ABC, abstractmethod
import keras
from tensorflow.keras.models import load_model
from pathlib import Path
import gc

class BaseModelLoader(ABC):
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None

        if not self.model_path.exists():
            raise FileNotFoundError(f'{self.model_path} does not exist')

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

class KerasModelLoader(BaseModelLoader):
    def __init__(self, model_path, custom_layers:dict=None, compile=False):
        super().__init__(model_path)
        self.custom_layers = custom_layers
        self.compile = compile
        self.custom_objects=custom_layers

    def load(self):
        if self.model is None:
            self.model=load_model(self.model_path, custom_objects=self.custom_objects, compile=self.compile)
        return self.model

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None

        keras.backend.clear_session()
        gc.collect()

class LiteRTModelLoader(BaseModelLoader):
    def __init__(self, model_path, num_threads=None):
        super().__init__(model_path)
        self.num_threads = num_threads

    def load(self):
        if self.model is None:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite import Interpreter

            self.model=Interpreter(model_path=self.model_path, num_threads=self.num_threads)
            self.model.allocate_tensors()

        return self.model

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()

class ModelLoaderFactory:

    @staticmethod
    def create(model_path, custom_layers:dict=None, num_threads:int = None):
        path = Path(model_path)
        ext=path.suffix.lower()

        if ext=='.tflite':
            return LiteRTModelLoader(path, num_threads=num_threads)
        elif ext in ['.h5', '.keras']:
            return KerasModelLoader(path, custom_layers=custom_layers)
        else:
            raise ValueError(f'Invalid model type: {ext}')

