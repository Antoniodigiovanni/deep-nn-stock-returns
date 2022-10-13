
echo Running the network results test
wget -O data.zip --no-check-certificate "https://onedrive.live.com/download?cid=9053A48EF4F6502C&resid=9053A48EF4F6502C%21128&authkey=ALogyVRe180RboQ"
unzip data.zip
pip install nni
#pip install matplotlib
#pip install pandas
#pip install numpy
pip install statsmodels


# Installation with CPU-only - comment the following line if a GPU is present
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# If a GPU is present in the system, use this command instead:
pip3 install torch torchvision torchaudio

python src/network_results_testing.py

