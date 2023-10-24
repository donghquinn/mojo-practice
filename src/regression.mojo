from python import Python


fn main() raises:
    let pd: PythonObject = Python.import_module("pandas")
    let np: PythonObject = Python.import_module("numpy")
    let stats: PythonObject = Python.import_module("statsmodels.api")
    let plt: PythonObject = Python.import_module("matplotlib.pyplot")

    var data: PythonObject = pd.read_csv("2023_Samcheonpo.csv")

    data = data.fillna(0)

    data = data.set_index(data["Date"])

    print(data)

    plt.plot(data["Solar_Radiation"])
    plt.show()

    let model = stats.OLS(data, "Solar_Radiation ~ Rain").fit()

    print(model.summary())
    # var refined = data

    # data["Date"] = pd.to_datetime(refined["Date"])

    # print(refined)
