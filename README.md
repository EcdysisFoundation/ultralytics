
https://docs.ultralytics.com/guides/conda-quickstart/

`conda create --name ultralytics-ev python=3.11 -y`

If using .zsh on mac

`conda init zsh` then close and reopen shell

`conda activate ultralytics-ev`

Install for Cuda

`conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics scikit-learn`

Install for local dev, without Nvidia GPUs

`conda install -c pytorch -c conda-forge pytorch torchvision ultralytics`

Ultralytics uses a very large collection of libraries, including many common libraries like pandas and numpy. To avoid problems with Ultralytics, only use the libraries it includes. If other libraries are needed for some task, create a different environment to run those tasks seperately.
