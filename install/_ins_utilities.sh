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
	git apply ../${dir}_*.patch || true
}

# $@: package names
ins_deb_require() {
	dpkg-query --show "$@"
}

# $1: URL of file containing gzipped dataset
# $2: expected SHA-512 checksum of that file
ins_data_get() {
  url="$1"
  sha512sum="$2"
  filename="${url##*/}"
  unzipped="${filename%.gz}"
  if [ ! -f "$unzipped" ]; then
    if [ ! -f "$filename" ]; then
      wget "$url"
    fi
    sha512sum --check - <<END && gunzip "$filename"
$sha512sum $filename
END
  fi
}
