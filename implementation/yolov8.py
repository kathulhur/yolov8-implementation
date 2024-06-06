import pathlib
from ultralytics import YOLO
from .abstraction import FilePath
from . import abstraction

MODULE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / 'output'

class YOLOv8Model(abstraction.Model):
    """
        The YOLOv8 implementation of the `Model`
    """

    def __init__(self, model): # accepts the yolo object that it will encapsulate
        self.model: YOLO = model

    def infer(self, input_file_paths: FilePath) -> abstraction.InferenceResult:
        # the yolov8 model expects a single input file
        input_file_path = input_file_paths[0]
        
        # START: model inference logic
        results = self.model.predict(source=str(input_file_path), save=True, project=str(OUTPUT_DIR))

        save_dir = pathlib.Path(results[0].save_dir)
        prediction_files = [item for item in save_dir.iterdir() if item.is_file]
        prediction_output_file = prediction_files[0]

        # END: model inference logic

        return {
            'data': prediction_output_file, # the file path of the output
            'info': {'any': 'info'}, # any information related to the inference result
            'type': "image/png" # MIME type of the output
        }

class YOLOv8ModelBuilder(abstraction.ModelBuilder):
    """
        The model builder builds a Model class

    """
    def __init__(self, model_artifacts_paths: list):
        """
            model_file_paths: the model artifacts
        """
        self.model_file_paths = model_artifacts_paths
    
    def build(self, model_file_paths: FilePath) -> abstraction.Model:
        # the model builder expects only a single file artifact
        model_weights = model_file_paths[0]

        yolo = YOLO(str(model_weights)) # the yolov8 inferencing class is instantiated

        return YOLOv8Model(yolo) # yolo is passed to the model for encapsulation



builder_class = YOLOv8ModelBuilder

inference_metadata = {
    'input_files': [ ['image'] ],
    'model_artifacts': [ ['.pt'] ]
}
