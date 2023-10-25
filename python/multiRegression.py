import time
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols


def multiRegression(
        fileName: str, 
        ):
    start = time.time()
  
    data = pd.read_csv(fileName)
    revised = data.fillna(0)
    
    print("Data Size: {}".format(len(revised)))
    
    test_size = round(len(data)*0.7)
    train, test = revised[test_size:], revised[:test_size]


    print("Start Multi Regression")
    
    formula = "solar_radiation ~ rain * temperature * wind_speed * wind_direction * visibility * total_cloud * pressure * humidity"
    
    model = ols(formula, train).fit()
    
    print("ADJ R2: {}".format(round(model.rsquared_adj, 3)))
    
    pred = model.predict(test)
    pred = pred.apply(lambda x: 0 if x < 0 else x)
    
    print("Finished Multi Regression")

    end = time.time()
    
    print('Elapsed Time: {}'.format(round(end - start, 3)))
    
    return pred

# oneSampleT("oneSampleTTest.xls", "용량", 250)
