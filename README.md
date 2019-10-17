# deephub
General purpose experimentation framework using Tensorflow and Keras

### Install
The code base works is compatible with >= Python 3.6 but it is *highly* proposed to stick on Python 3.6 
and something bigger. It is expected that interpreter is installed before continuing with this guide. 

It is proposed to use python virtual virtual environments to handle different projects. To create and activate a 
virtual environment for python 3 you need to run:

```
$ python3.6 -m venv venv
$ source venv/bin/activate
```
Make sure you have updated your setuptools and pip versions using:
```
$ pip install --upgrade setuptools pip
```
Next you have to install package dependencies, cli tool using the pip command. If you work on development environment
it is prefered to use `-e` option:
```
$ pip install -e .
```

### Code Base Usage

#### Code Base Entry Points
DeepResearch framework provides different cli entry points in order to be able to use different modules from the
framework. These entry points are `{resources, trainer, hpt, utils}` with the prefix of `deep` command which is the
entry point for the whole framework.

### Resources
[Resources System](docs/Resources.md)
### Training
[Training Procedure](docs/Training.md)
### Testing
[Testing Procedure](docs/Testing.md)

### Important notes ###
#### NLTK models
Some functionality related with text processing depends on NLTK's external resources to work properly. After
installation of requirements you need to manually download these resources:

```
$ python -m nltk.downloader punkt
```

#### GPU experiments
In order to be able to use this framework on DeepLab's servers with GPUJ support you have to add these lines:
```
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
export PATH=/usr/local/cuda-9.0/bin:$PATH
```
in the end of your `.bashrc` and then run `source ~/.bashrc`.

