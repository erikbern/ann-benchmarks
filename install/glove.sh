cd "$(dirname "$0")"
wget "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz"
gzip -d glove.twitter.27B.100d.txt.gz
cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt # strip first column
rm glove.twitter.27B.*
