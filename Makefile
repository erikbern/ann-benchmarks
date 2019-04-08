.PHONY: default
default:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

.PHONY: install
install:
	python install.py --algorithm=nann

.PHONY: build
build:
	fpie/fpie.sh .. .includefile | docker build --rm -t ann-benchmarks-nann -f ann-benchmarks/install/Dockerfile.nann -

.PHONY: test
test:
	python run.py --algorithm=nann

.PHONY: both
both: build test
