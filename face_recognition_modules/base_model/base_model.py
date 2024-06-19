from easydict import EasyDict as edict
import onnxruntime
import threading

class BaseModel:
    def __init__(self, model_path, device='cpu', **kwargs) -> None:
        self.model = self.load_model(model_path, device)
        self.input_layer = self.model.get_inputs()[0].name
        self.output_layers = [output.name for output in self.model.get_outputs()]
        self.lock = threading.Lock()

    def load_model(self, model_path:str, device:str='cpu'):
        available_providers = onnxruntime.get_available_providers()
        if device == "gpu" and "CUDAExecutionProvider" not in available_providers:
            print("CUDAExecutionProvider is not available, use CPUExecutionProvider instead")
            device = "cpu"

        if device == 'cpu':
            self.model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            self.model = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
            
        return self.model
        
    def inference(self, input):
        with self.lock:
            outputs = self.model.run(self.output_layers, {self.input_layer: input})
        return outputs
        
    def preprocess(self, **kwargs):
        pass

    def postprocess(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass