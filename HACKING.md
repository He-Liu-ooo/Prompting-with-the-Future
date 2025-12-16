# HACKING
## ENV Set Up
CUDA version: 11.8
pytorch3d: conda install pytorch3d::pytorch3d -c pytorch3d -c pytorch -c conda-forge
SAM2: download README.md from https://github.com/facebookresearch/sam2/blob/main/README.md and upload it under sam2/ before pip install -e .
COLMAP: easy install a usable version first
```
conda install -c conda-forge colmap
colmap -h # make sure runnable
```
When running main.py, a lot of missing packages reported
e3nn: pip install "e3nn==0.5.1"
downgrade numpy: 
```
pip uninstall -y numpy
pip install "numpy==1.26.4"
```

## Run
Encoding images to base64 b is crazy since this will expand the prompt tokens to an exploded number