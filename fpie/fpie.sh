#!/usr/bin/env bash
set -eo pipefail

usage_help () {
  echo "FPIE packs up a context based a path and an include/exclude-file."
  echo "Usage: fpie.sh CONTEXT_SRC INCLUDEFILE"
  echo ""
  echo "SYNTAX"
  echo "Files and folders are defined using paths relative to the context path."
  echo "You can either include files:"
  echo "  includefile"
  echo "Or all files within a folder:"
  echo "  includedir/*"
  echo "Specifying a folder with a trailing '/' does not work."
  echo "Prepend a filepath with '!' for exclusion:"
  echo "  !excludefile"
  echo "Or exclude an entire directory:"
  echo "  !excludedir"
  echo "When excluding a directory inside an included directory, all files"
  echo "in the excluded directory need to be excluded explicitly:"
  echo "  !**/excludedir/*"
  echo "  !**/excludedir"
  echo "It is also possible to exclude using glob patterns:"
  echo "  !**/*.pyc"
  echo "Comments need to be on a separate line:"
  echo "  #This is a comment" 
  echo "  includefile"
  echo "  excludefile"
}

if [[ $# -lt 2 ]]; then
  usage_help
  exit 2
elif [[ $# -gt 2 ]]; then
  usage_help
  exit 2
fi

SCRIPTLOCATION=`dirname $0`
# Run both the filefinder and filepacker, to produce an
# include/excluded tar.
${SCRIPTLOCATION}/_filefinder.sh $1 $2 | ${SCRIPTLOCATION}/_filepacker.sh $1

