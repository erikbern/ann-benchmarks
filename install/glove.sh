cd "$(dirname "$0")"
wget "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
unzip glove.twitter.27B.zip
cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt # strip first column
rm glove.twitter.27B.*
