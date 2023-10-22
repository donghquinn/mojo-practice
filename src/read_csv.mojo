from python import Python


fn main() raises:
    Python.add_to_path("./python")

    let pd: PythonObject = Python.import_module("pandas")
    let data: PythonObject = pd.read_csv("read.csv")

    print(data)
