## Ultralytics and SAHI

This repo tracks code we use with Ultrlaytics primarly for image segmentation using Slicing Aided Hyper Inference for large panoramas. (see https://github.com/obss/sahi ). These panoramas are generated on our Stitcher api (see https://github.com/EcdysisFoundation/stitcher ). Dataset generation and inference uses this api to access images. We use segmentation to identify arthropods in the panorama, and we then in later proceses later crop those out and run our classification model on them, using https://github.com/EcdysisFoundation/metaformer_ecdysis .

### The environment

https://docs.ultralytics.com/guides/conda-quickstart/

1. Create a fresh environment while forcing NumPy 1.x compatibility from the start

`conda create --name ultralytics_n1 python=3.11 "numpy<2" -y`

2. Activate the new environment

`conda activate ultralytics_n1`

3. Install core CUDA, PyTorch, and heavy framework binaries via Conda

```
conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch \
    torchvision \
    pytorch-cuda=12.4 \
    ultralytics \
    scikit-learn \
    scikit-image \
    pycocotools
```

4. Safely upgrade or install SAHI using pip without altering the core NumPy version

`pip install --upgrade sahi --no-deps`

5. Avoid system binary conflicts for this install

`pip install shapely --no-build-isolation`

6. Install SAHI's missing CLI and progress utility packages

`pip install click fire tqdm`

7. Problems with 2024 MKL backend???

`conda install -y -c conda-forge "mkl<2024.1"`

8. Optional minor pip installations go last?
`pip install pyyaml`

[!TIP]
Ultralytics uses a very large collection of libraries, including many common libraries like pandas and numpy. Integrations like SAHI may require many specific configurations as seen above. To avoid problems with Ultralytics, try to only use the libraries it includes and the minimum needed to run SAHI.

### Symlink image files

We use symlinks to access images over a local private network using NFS.

Can check if the entry still exists by viewing filesystem usage with `df -H`

### Training Dataset Generation

We annotate our panorama images with cvat.ai and export those annotations as YOLO .txt files, one per image. These are structured in our Stitcher FastAPI database (https://github.com/EcdysisFoundation/stitcher).

Run the dataset generation script to convert those to coco, slice, and structure as a YOLO training dataset.

`python -m dataset_generation`

Label-studio is also compatible, export annotations from label-studio using json-min format and use option `--label-platform label-studio`

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
