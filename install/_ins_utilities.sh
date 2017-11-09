# ($0: target directory)
# $1: URL
# (changes directory to target directory)
ins_git_get() {
  dir="${0##*/}"
  dir="${dir%%.sh}"
  if [ ! -d "$dir" ]; then
    git clone "$1" "$dir"
  else (
    cd "$dir" &&
    git clean -dfx &&
    git reset --hard origin/master &&
    git pull
  ) fi
  cd "$dir" &&
  git am ../${dir}_*.patch || true
}

# $@: package names
ins_deb_require() {
  dpkg-query --show "$@" || apt-get install -y "$@"
}

ins_pip_get() {
  ins_deb_require python-pip && pip install "$@"
}

# $1: URL of file containing gzipped dataset
# $2: expected MD5 checksum of that file
ins_data_get() {
  url="$1"
  md5sum="$2"
  filename="${url##*/}"
  unzipped="${filename%.gz}"
  if [ ! -f "$unzipped" ]; then
    if [ ! -f "$filename" ]; then
      wget "$url"
    fi
    md5sum --check - <<END && gunzip "$filename"
$md5sum $filename
END
  fi
}
