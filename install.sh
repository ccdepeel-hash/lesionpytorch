# environment creation - GPU installation script
# Create environment with all packages
conda create -n plantlesion_gpu2 python=3.11 pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=12.1 "numpy<2" matplotlib pillow -c pytorch -c nvidia -y

# Activatation
conda activate plantlesion_gpu2

# Verification
python -c "import torch; import torchvision; import numpy as np; import matplotlib; from PIL import Image; print(f'Python: {torch.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {np.__version__}'); print(f'Matplotlib: {matplotlib.__version__}'); print(f'Pillow: {Image.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# vvv environment creation - CPU installation script vvv
# conda create -n plantlesion_cpu python=3.11 pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 cpuonly "numpy<2" matplotlib pillow -c pytorch -y

# Activatation
# conda activate plantlesion_cpu

# Verification
# python -c "import torch; import torchvision; import numpy as np; import matplotlib; from PIL import Image; print(f'Python: {torch.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {np.__version__}'); print(f'Matplotlib: {matplotlib.__version__}'); print(f'Pillow: {Image.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"