cd "$(dirname "$0")"
wget "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
tar -xzf sift.tar.gz
rm -rf sift.tar.gz
wget "https://raw.githubusercontent.com/searchivarius/NonMetricSpaceLib/master/data/data_conv/convert_texmex_fvec.py"
python convert_texmex_fvec.py sift/sift_base.fvecs >> sift.txt
rm -rf sift

