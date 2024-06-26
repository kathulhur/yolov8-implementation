from implementation import builder_class
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent
model_artifact_path = ROOT_DIR / 'implementation' / 'assets' / 'yolov8n.pt'
sample_image = ROOT_DIR / 'implementation' / 'assets' / 'sample-image.png'

builder = builder_class()
model = builder.build([str(model_artifact_path)])
result = model.infer([str(sample_image)])

