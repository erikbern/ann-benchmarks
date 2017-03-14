#!/bin/sh
cd "$(dirname "$0")"
. ./_ins_utilities.sh

sh lib-annitu.sh --just-get > /dev/null 2>&1 || exit 1
if [ ! -f "webspam-data.txt" ]; then
  cp lib-annitu/datasets/webspam/data_256.txt webspam-data.txt &&
    cat > webspam-data.yaml <<END
dataset:
  point_type: bit
END
fi

if [ ! -f "webspam-query.txt" ]; then
  cp lib-annitu/datasets/webspam/query_256.txt webspam-query.txt &&
    cat > webspam-query.yaml <<END
dataset:
  point_type: bit
END
fi
