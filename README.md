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

## L2 numpy and qubit computing
>Numpy
> - Illustrated tutorial and 100 exercises show some interesting examples of advance application of numpy array

>Classes

## L3 optimization, profiling, and random simulation
>Optimization
> - There are different strategies to improve the performances. E.g., Numba (njit), functools (lru_cache), einsum

>Profiling with iPython
> ```
> %timeit function
> ```
> ```
> %%timeit
> ...
> ```
> ```
> %load_ext line_profiler
> %prun function(a)
> %prun -f sub_function function(a)
> ```

>Profiling with Austin and VSCode
> - Install the extensions `Austin VS Code` https://marketplace.visualstudio.com/items?itemName=p403n1x87.austin-vscode and `Flame Chart Visualizer for JavaScript Profiles` https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-js-profile-flame
> - Install the conda package `conda install -c conda-forge austin` or `pip install austin-dist`
> - Create a script e.g. `test_l3.py` that runs the code you want to profile (both cpu and memory!). Pay attention to the length of the script because the the output file length grows really fast
> - Run austin profiler (there should be a way directly from vscode, but it doesn't work for me, so I do it from the console) with `austin -i 100 --pipe C:\Users\davtom\AppData\Local\miniforge3\envs\sciqis\python.exe ".\test_l3.py" > test.austin` (Use the python of the conda environment) (Look online for all the options https://github.com/P403n1x87/austin-vscode?tab=readme-ov-file)
> - The output is in the file test.austin, to open it use `FlameGraph` tab in vscode (shortcut `Win/Cmd+Shift+A`) and select the file. You can navigate all the functions in the code, see the tree of the calls, and the total cumulative time if the same function is called multiple times.

>Random simulation with Numpy
> ```
> rng = np.random.default_rng()
> rng.uniform(0, 1, size=(10,2))