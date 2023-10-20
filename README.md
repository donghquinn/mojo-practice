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