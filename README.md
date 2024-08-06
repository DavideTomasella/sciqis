# Scientific Computing in Quantum Information Science - Summer 2024
## L1 environment sciqis and package
 >Setup environments
 > - install miniforge in the local user
 > - `mamba` and `conda` are equivalent
 > - create the env from the file `mamba create -f "env-config.yml"` it will add all the needed packages
 > - when possible use `mamba install package` to add the packages so the channel is the same
 > - `mamba activate sciqis`, `mamba deactivate`
 > - `conda info` and `conda info --system`
 > - export with `conda env export -n sciqis > env-config.yml`
 > NOTE: with qt, rpyc and ... you must use pip channel instead of the default mamba (see qudi project)
 >       You can create env variables (`conda env config vars list` and `conda env config vars set HI="hello"`)

 >Create package
 > - Inside the dir, add `src>mylib` with the lib name. Here create `__init__.py` with the relative  import of the files that must be seen from the user `from .myclass import *` (use relative path to init  file)
 > - To install a develop package we need to use pdm to define all the configs
 > - So in the main folder we have `pip install pdm`, `pdm init`, and `pdm install`
 > - Follow the instruction to add dependences `pdm add dependence_package`
 > - Then within your conda env `pip install -e ".\mylib_package"` to add the package in dev mode: no  need to reinstall after each change
 > - To use it, `from mylib.myclass import MyClass2` so specify the file name with `src` and the class/method name