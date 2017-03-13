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
