from python import Python


def main():
    Python.add_to_path("python_modules/lstm")

    let model = Python.import_module("lstm_stock")

    model.stock()
