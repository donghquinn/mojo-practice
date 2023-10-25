from python import Python


fn main() raises:
    let pd: PythonObject = Python.import_module("pandas")
    let np: PythonObject = Python.import_module("numpy")
    let stats: PythonObject = Python.import_module("statsmodels.api")
    let plt: PythonObject = Python.import_module("matplotlib.pyplot")

    var data: PythonObject = pd.read_csv("combined.csv")

    data = data.fillna(0)

    data = data.set_index(data["date"])

    data = data[
        [
            "temperature",
            "rain",
            "humidity",
            "visibility",
            "pressure",
            "wind_speed",
            "wind_direction",
            "total_cloud",
            "solar_radiation",
        ]
    ]

    print(data)

    let formula = "solar_radiation~rain*temperature*humidity*total_cloud*pressure*visibility"
    var model = stats.OLS(data, formula)

    model = model.fit()

    print(model.summary())
