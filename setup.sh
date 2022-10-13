
echo \n\n\nPreparing data for experiments and running a first test without NNI\n\n\n
wget -O data.zip --no-check-certificate "https://onedrive.live.com/download?cid=9053A48EF4F6502C&resid=9053A48EF4F6502C%21128&authkey=ALogyVRe180RboQ"
sudo apt update
sudo apt install unzip
unzip data.zip
sudo apt install python3-pip
pip install nni
#pip install matplotlib
#pip install pandas
#pip install numpy
pip install statsmodels

pip install cloud-tpu-client==0.10 torch==1.12.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl || pip3 install cloud-tpu-client==0.10 torch==1.12.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl

# Installation with CPU-only - comment the following line if a GPU is present
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# If a GPU is present in the system, use this command instead:
# pip3 install torch torchvision torchaudio

python src/network_training_test.py || python3 src/network_training_test.py

