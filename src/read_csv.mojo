from python import Python


fn main() raises:
    let pd: PythonObject = Python.import_module("pandas")
    let data: PythonObject = pd.read_csv("datasets/read.csv")

    print(data)

    print(data.describe())
