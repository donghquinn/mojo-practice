import os
from ultralytics import YOLO

def yoloV8():
    model = YOLO()

    model.train(data=os.path.join("python_modules/data.yml"), epochs=20)

    model.save()

# yoloV8()