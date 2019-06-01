## Install fairseq utils (Not needed if regular training script is not to be run)
#cd `dirname $0`
#pip install --editable .

# Download dataset (Downloading smaller dataset only)
mkdir ~/data/
cd ~/data
wget https://dl.fbaipublicfiles.com/fairseq/data/wmt18_en_de_bpej32k.zip && unzip wmt18_en_de_bpej32k.zip
#wget https://dl.fbaipublicfiles.com/fairseq/data/wmt18_en_de_bpej32k_btdata.zip && unzip wmt18_en_de_bpej32k_btdata.zip
#mv wmt18_en_de_bpej32k/* wmt18_en_de_bpej32k_btdata/
