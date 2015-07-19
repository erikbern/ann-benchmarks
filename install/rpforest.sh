cd "$(dirname "$0")"
apt-get install -y python-dev python-setuptools
git clone https://github.com/lyst/rpforest
cd rpforest
python setup.py install
cd ..
