from python import Python


fn main() raises:
    Python.add_to_path("python_modules/lstm")

    let model = Python.import_module("lstm_stock")

    let result = model.stock()

    print(result)
