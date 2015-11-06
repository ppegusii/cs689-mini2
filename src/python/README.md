# cs689-mini2
## CMPSCI-689 mini project 2
## Description
Play with distance measures between words embedded in a vector space.
A comparison of various machine learning methods to classify a person by their characteristic walking gate.
## Building
Using conda for package management. [Here](http://conda.pydata.org/docs/using/envs.html) is a basic tutorial.
### Create environment from file
`conda env create -f environment.yml`
### Update environment from file
`conda env update -f environment.yml`
### Activate the environment
Linux, OS X: `source activate mini1`

Windows: `activate mini1`
### Deactivate the environment
Linux, OS X: `source deactivate mini1` or `source deactivate`

Windows: `deactivate mini1`
### Install a new package
Make sure mini1 is the active environment.

`conda install package_name`
### Export the environment to a file
Make sure mini1 is the active environment.

`conda env export > environment.yml`
