from python import Python


fn main() raises:
    let pd: PythonObject = Python.import_module("pandas")
    let np: PythonObject = Python.import_module("numpy")

    let arrays = np.array([1, 2, 3, 2, 3, 4, 4, 5, 6]).reshape(3, 3).T

    # DataFrame(data, index, columns)
    let df = pd.DataFrame(arrays, np.arange(3), ["cola", "colb", "colc"])

    print(df)
