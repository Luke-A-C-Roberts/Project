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
wsl --install # for me it throws an error without the --install flag even if installed
```

In WSL (install python dependancies)
```bash
conda install cudatoolkit cudnn # make sure there is a gpu available
export LD_LIBRARY_PATH=$LD_LIBRARY:$CONDA_PREFIX/lib
pip3 -m pip install tensorflow[and-gpu] findspark pyspark pandas matplotlib seaborn
```

In WSL (install data)
```bash
# preparation
sudo apt install unzip
mkdir ~/Downloads/
cd Downloads/

# downloading files
wget https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz
wget https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1
wget https://zenodo.org/records/3565489/files/images_gz2.zip?download=1 # this wget takes some time

# renaming
mv 'gz2_filename_mapping.csv?download=1' gz2_filename_mapping.csv
mv 'images_gz2.zip?download=1' images_gz2.zip

# extracting
gzip -d gz2_hart16.csv.gz
unzip images_gz2.zip

# cleanup
rm __MACOSX/ -rf
rm images_gz2.zip
```ls
