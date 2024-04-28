# To build

In windows CMD:
```bash
wsl --install
```

In WSL (Insall Conda):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
exit
```

Re-enter WSL:
```bash
wsl --install
```

In WSL (install python dependancies)
```bash
conda install cudatoolkit cudnn
export LD_LIBRARY_PATH=$LD_LIBRARY:$CONDA_PREFIX/lib
pip3 -m pip install tensorflow[and-gpu] findspark pyspark pandas matplotlib seaborn
```

In WSL (install data)
```bash
sudo apt install unzip


```
