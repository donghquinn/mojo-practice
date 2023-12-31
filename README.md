# [MOJO install on MAC Apple Silicon](https://developer.modular.com/download)

## Install Modular CLI

``` 
curl https://get.modular.com | sh - && \
modular auth mut_3bf7c0beb0904edabf64333e3beb94d8 
```

## Install Mojo SDK

```
modular install mojo
```

## Install mojo extension on VS Code(Optional)


## For ZSH users

```
echo 'export MODULAR_HOME="/Users/<USER_NAME>/.modular"' >> ~/.zshrc
echo 'export PATH="/Users/<USER_NAME>/.modular/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Check
```
mojo
```

## File Structures

### src
- mojo files
    - list: List and Print Test
    - dataframe: Numpy List test & Pandas Dataframe Test
    - read_csv: Pandas Read CSV Test
    - system: Check System Information
    - regression: StatsModels OLS Multi Regression Test (WIP)
    - lstm: Long Short-Term Memory Test & Import Python Custom Functions

- python modules
    - lstm: LSTM Model
    - stock: Stock Prediction Functions
    - visualize: Stock Prediction Visualization
    - multiRegression: Multi Regression with statsmodels

- stock.csv: CSV Dataset with LSTM
- combined.csv: Multi Regression Dataset (South Korean KMA ASOS API)
- estimate.csv: LSTM Prediction Estimated Score