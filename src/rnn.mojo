from python import Python


def main():
    Python.add_to_path("python_modules/rnn")

    let model = Python.import_module("rnn_stock")

    model.stock()
