from python import Python


fn main() raises:
    Python.add_to_path("python_modules/rnn")

    let model = Python.import_module("rnn_stock")

    let result = model.stock()

    print(result)
