# environment creation
conda create -n plantlesion python=3.9 numpy=1.26.4
conda activate plantlesion
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
# (plantlesion) should appear. 
# python 3.9, numpy 1.26.4 used for pytorch compatibility
# to check versions: python -c "import numpy; print(numpy.__version__) | print(torch.__version__)| torch 2.2.2
# matplotlib used to display predicted images in predict.py