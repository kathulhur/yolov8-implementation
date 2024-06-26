import pathlib, mimetypes
from ultralytics import YOLO
from .abstraction import FilePath
from . import abstraction
from moviepy.editor import VideoFileClip


MODULE_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / 'output'

class YOLOv8Model(abstraction.Model):
    """
        The YOLOv8 implementation of the `Model`
    """

    def __init__(self, model): # accepts the yolo object that it will encapsulate
        self.model: YOLO = model

    def infer(self, input_file_paths: list[str]) -> abstraction.InferenceResult:

        # the yolov8 model expects a single input file
        input_file_path = input_file_paths[0]

        mimeType, encoding = mimetypes.guess_type(input_file_path)
        
        mediaType = mimeType.split('/')[0]

        if mediaType == 'image':
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
        
        elif mediaType == 'video':
            # START: model inference logic
            results = self.model.predict(source=str(input_file_path), save=True, project=str(OUTPUT_DIR))

            save_dir = pathlib.Path(results[0].save_dir)
            prediction_files = [item for item in save_dir.iterdir() if item.is_file]
            prediction_output_file = prediction_files[0]

            prediction_output_file_mp4_path = prediction_output_file.with_suffix('.mp4')
            # Load the video file
            clip = VideoFileClip(str(prediction_output_file))
            
            # Write the video file to MP4 format
            clip.write_videofile(str(prediction_output_file_mp4_path), codec="libx264")

            # END: model inference logic

            return {
                'data': str(prediction_output_file_mp4_path), # the file path of the output
                'info': {'any': 'info'}, # any information related to the inference result
                'type': "video/mp4" # MIME type of the output
            }

class YOLOv8ModelBuilder(abstraction.ModelBuilder):
    """
        The model builder builds a Model class

    """
    
    def build(self, model_file_paths: list[str]) -> abstraction.Model:
        # the model builder expects only a single file artifact
        model_weights = model_file_paths[0]
        yolo = YOLO(model_weights) # the yolov8 inferencing class is instantiated

        return YOLOv8Model(yolo) # yolo is passed to the model for encapsulation



model_builder_class = YOLOv8ModelBuilder

inference_metadata = {
    'input_files': [ ['image/*', 'video/*'] ], # requires a single input file with an image type
    'model_artifacts': [ ['.pt'] ] # requires one model artifact with a .pt file extension
}
