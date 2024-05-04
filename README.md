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

In WSL install python dependancies
```bash
conda install cudatoolkit cudnn # make sure there is a gpu available
export LD_LIBRARY_PATH=$LD_LIBRARY:$CONDA_PREFIX/lib
python3 -m pip install tensorflow[and-gpu] findspark pyspark pandas scikit-learn matplotlib seaborn more-itertools
python3 -c "import tensorflow as tf; print('\n'.join([*map(str, tf.config.list_physical_devices())]))"
```

In WSL install data
```bash
mkdir ~/Downloads/
cd Downloads/

wget https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz
wget https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1

mv 'gz2_filename_mapping.csv?download=1' gz2_filename_mapping.csv

gzip -d gz2_hart16.csv.gz
```

In WSL install raw images
```bash
sudo apt install unzip

wget https://zenodo.org/records/3565489/files/images_gz2.zip?download=1
unzip images_gz2.zip

mv 'images_gz2.zip?download=1' images_gz2.zip

cd
```

In WSL install preprocessed images (alternatively)
```bash
pip install gdown
gdown --id 1KwKv5uFMuF-wHa_7JcF3oNc16CG-8tjz --output preprocessed.tgz

tar -zxvf preprocessed.tgz

cd
```

In WSL download this repo
```bash
git clone https://github.com/Luke-A-C-Roberts/Project
```

In WSL Apply Preprocessing (optional)
```bash
cd Project/Software/Final/
python3 preprocessing.py
cd
```



Add a decent editor (optional)
```bash
sudo add-apt-repository ppa:maveonair/helix-editor
sudo apt update
sudo apt install helix
```
