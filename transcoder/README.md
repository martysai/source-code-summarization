# TransCoder

To set up conda environment for this subfolder:
```
conda create -n transcoder python=3.7
conda install pytorch torchtext cudatoolkit=10.2 -c pytorch
```

Then installing `fastBPE`:
```
cd transcoder/XLM
bash install-tools.sh
```
