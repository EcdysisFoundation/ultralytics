
### The environment

https://docs.ultralytics.com/guides/conda-quickstart/
Using alternative approaches known to cause problems. This works on Unbuntu and Nvidia's Cuda. Pip versions may not.

`conda create --name ultralytics-ev python=3.11 -y`

If using .zsh on mac

`conda init zsh` then close and reopen shell

Activatae environment

`conda activate ultralytics-ev`

Install for Cuda

`conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics scikit-learn`

Install for local dev, without Nvidia GPUs

`conda install -c pytorch -c conda-forge pytorch torchvision ultralytics sckit-learn`

Ultralytics uses a very large collection of libraries, including many common libraries like pandas and numpy. To avoid problems with Ultralytics, try to only try to use the libraries it includes and those included above. If any other libraries are added, they need to be tested and documented here.

Add library for SAHI: Slicing Aided Hyper Inference. Known to be compatibale with YOLO.

`conda install sahi scikit-image -c conda-forge`

### Symlink image files

The images are accessed through symlinks created during dataset generation. The drive location on Ecdysis01 needs to be mapped for this to work. This is done with the following command, and will need to be re-linked after a system reboot. For new mounts, first make a directory locally where the external mount will exist (second argument).

bugbox3

`sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media/ -o allow_other`

stitcher

`sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/label-studio/mydata/stitchermedia /pool1/srv/label-studio/mydata/stitchermedia -o allow_other`

Can check if the entry still exists by viewing filesystem usage with `df -H`


### Dataset Generation

Using the conda environment, `conda activate ultralytics-ev`

Export annotations from label-studio using json-min format.

Run the dataset generation script

`python -m dataset_generation`

### Training

Run with output saved to file

`python -m train > last_training.log 2>&1 &`

To run on both GPU's, there will be an error due to Intel library incompatibility, but you can use the force variable

`export MKL_SERVICE_FORCE_INTEL=1`

To resolve `Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.`

### Deployment

zip the entire dir to download and examine output

    tar -zcvf OUTPUTDIR.tar.gz OUTPUTDIR

#### For inference with SAHI

Inference with SAHI requires the ultralytics library. Replace the MODEL_PATH to the model weights in inference.sahi_stitched. And run inference using the __main__.py in the inference module.

#### For inference without SAHI
The trained model is deployed using FastAPI. See https://github.com/EcdysisFoundation/inference-fastapi. Two files are required. 1. The data.yaml renamed to yolo_data.yaml and 2. the model exported to .onnx format, renamed to yolo_best.onnx.

To export the model, use for exmple

    model = YOLO("path/to/best.pt")  # load a custom trained model

    model.export(format="onnx")  # export to .onnx format
