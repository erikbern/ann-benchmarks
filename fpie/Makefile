IMAGE_NAME=siteimprove/docker-tools/fpie

.PHONY: default
default:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

.PHONY: docker-pack
docker-pack:
	docker build -t ${IMAGE_NAME} -f Dockerfile .

.PHONY: test
test:
	@cd tests \
	&& ./_filepacker_tests.sh \
	&& ./_filefinder_tests.sh \
	&& echo "\033[1;32mAll OK\033[0m" \
	|| echo "\033[1;31mTest failed\033[0m"
