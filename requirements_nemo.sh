conda create --name nemo python==3.8
conda activate nemo
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

apt-get install sox libsndfile1 ffmpeg
pip install wget
pip install text-unidecode
pip install Cython
pip install nemo_toolkit['all']