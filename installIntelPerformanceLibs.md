# Installing Intel Python Libraries

Intel provides an optimized version of Python.

This tutorial assumes you have installed miniconda. You can refer to [Installing Miniconda and Pytorch](installPyTorch.md) if you need some guidance.

I basically followed the instructions from:
[Installing Intel Distribution for Python](https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html)

## Pre-Req's

* Update Conda

    ```sh
    conda update conda && conda config --add channels intel
    ```

    The updates will vary depending on how many updates you need, but it will ask before performing the update. It should look something like:

    ```sh
    Proceed ([y]/n)?
    ```

    Accept and let it run. It may take some time for the updates to download.

* Add the intel channel to conda
  
    ```sh
    conda config --add channels intel
    ```

## Create an Intel Optimized environment

We can now create an intel optimized environment. In order to match our other environment (created in [InstallPyTorch](installPyTorch.md)) we will specify name `idp` and `python 3.8`. In the next command, the libraries we will install are in `intelpython3_core`:

```sh
conda create -n idp intelpython3_core python=3.8
```

Conda will run and create a `Package Plan`, look through the packages if you are interested, but you don't really need to know the details to work with the environment. Conda will ask if you wish to proceed:

```sh
Proceed ([y]/n)? 
```

Accept and let it run. It may take some time for the packages to download. Once it finishes, it should look something similar to:

```sh
# To activate this environment, use
#
#     $ conda activate idp
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

## Switch to the New idp Environment

We should now be able to switch to the new environment by typing:

```sh
conda activate idp
```

Verify your version of python with:

```sh
python --version
```

if everything worked, it should respond with something similar to:

```sh
python --version
Python 3.8.12 :: Intel Corporation
```

the minor version may be different, but the important part is that it returns `:: Intel Corporation`

## Install pytorch and other libraries

Install Pytorch
> Note: I had to add `torchvision` and `torchaudio` in order to get the tutorials section to work.

```sh
conda install pytorch torchvision torchaudio -c pytorch
```

conda will take some time to figure out what needs to be downloaded and prompt you before downloading:

```sh
Proceed ([y]/n)? 
```

Accept and let it run. It may take some time for the packages to download.

Install [additional libraries from Jeff](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/tools.yml) which may be helpful.

```sh
# get the file that contains the list of tools
wget https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/tools.yml
# now install them
conda env update --file tools.yml
```

conda will take some time to figure out what needs to be downloaded and prompt you before downloading:

```sh
Proceed ([y]/n)? 
```

Accept and let it run. It may take some time for the packages to download.

## Conclusion

You should now have your environment set up and ready to go.
I've included several other tutorials in the [README.md](README.md) file if you want to learn more.

I can be reached at [@IAmDanielV](https://twitter.com/iamdanielv) on Twitter if you have any questions or suggestions.

Thanks!
