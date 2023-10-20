from python import Python


fn main() raises:
    Python.add_to_path("./python")

    let pd: PythonObject = Python.import_module("pandas")
    let read = Python.import_module("read_csv")
    let data = read.read_csv("read.csv")

    print(data)
