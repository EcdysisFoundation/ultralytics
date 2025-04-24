
### The environment

https://docs.ultralytics.com/guides/conda-quickstart/
Using alternative approaches known to cause problems.

`conda create --name ultralytics-ev python=3.11 -y`

If using .zsh on mac

`conda init zsh` then close and reopen shell

`conda activate ultralytics-ev`

Install for Cuda

`conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics scikit-learn`

Install for local dev, without Nvidia GPUs

`conda install -c pytorch -c conda-forge pytorch torchvision ultralytics sckit-learn`

Ultralytics uses a very large collection of libraries, including many common libraries like pandas and numpy. To avoid problems with Ultralytics, try to only try to use the libraries it includes and those included above. If any other libraries are added, they need to be tested and documented here.

### Symlink image files

The images are accessed through symlinks created during dataset generation. The drive location on Ecdysis01 needs to be mapped for this to work. This is done with the following command, and will need to be re-linked after a system reboot.

`sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media/ -o allow_other`

Can check if the entry still exists by viewing `proc/self/mounts` as seen below. Or on filesystem usage with `df -H`

`ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media fuse.sshfs rw,nosuid,nodev,relatime,user_id=0,group_id=0,allow_other 0 0`

### Dataset Generation

Download a new training selections file to Ecdysis02

`scp ecdysis@ecdysis01.local:/pool1/srv/bugbox3/local_files/obj_det_selections.json ./local_files/obj_det_selections.json`

Run the dataset generation script

`python -m dataset_generation`


