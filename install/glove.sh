wget "http://www-nlp.stanford.edu/data/glove.twitter.27B.100d.txt.gz"
gzip -d glove.twitter.27B.100d.txt.gz
cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt # strip first column
