from python import Python


def main():
    Python.add_to_path("python")

    let model = Python.import_module("main")

    model.stock_predict()
