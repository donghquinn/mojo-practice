from python import Python

def main():
    Python.add_to_path("python_modules")
    app = Python.import_module("yolo_v8")

    app.yoloV8()
