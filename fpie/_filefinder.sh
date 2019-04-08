#!/usr/bin/env bash
set -eo pipefail

IFS=""

# Declare quiet pushd command
quiet_pushd () {
    command pushd "$@" > /dev/null
}

# Declare quiet popd command
quiet_popd () {
    command popd "$@" > /dev/null
}

# Reset pwd on exit
trap quiet_popd EXIT

# Declare command builder
build_command () {
    PATTERNS=("$@")
    # Keep track of first pattern
    IS_FIRST_FIND=0
    FINDCOMMAND="find -type f"

    # use quotes to avoid expansion of file patterns
    for pattern in "${PATTERNS[@]}"; do
        # Append to the find command.
        # First append is a bit special.
        if [[ ${IS_FIRST_FIND} = 0 ]] ;
        then
            IS_FIRST_FIND=1
            FINDCOMMAND="$FINDCOMMAND -path \"./${pattern}\""
        else
            FINDCOMMAND="$FINDCOMMAND -o -path \"./${pattern}\""
        fi
    done

    echo $FINDCOMMAND
}

# Grab the paths of the two arguments coming in.
FILECONTEXT=`realpath $1`
INCLUDEFILE=`realpath $2`

# Go to the context from where to get files
quiet_pushd ${FILECONTEXT}

# Enable globbing expansion.
shopt -s globstar

# Pattern arrays
INCLUDEPATTERNS=()
EXCLUDEPATTERNS=()

# Loop over all lines in the include file.
while read -r line || [[ -n "${line}" ]]
do
    # Ignore comments and empty lines
    if [[ ${line} = \#* || -z "${line}" ]] ;
    then
        : # Do nothing
    # Lines with '!' in front are excludes.
    elif [[ ${line} = \!* ]] ;
    then
        # leave out first char in front of exclude patterns
        EXCLUDEPATTERNS+=("${line:1}")
    else
        INCLUDEPATTERNS+=("${line}")
    fi
done < ${INCLUDEFILE}

# if there is no positive match, exit
if [[ ${#INCLUDEPATTERNS[@]} = 0 ]] ;
then
    exit 0
fi

INCLUDEFINDCOMMAND=$(build_command "${INCLUDEPATTERNS[@]}")

FINALFILELIST=()
if [[ ${#EXCLUDEPATTERNS[@]} = 0 ]] ;
then
    # If there is no negative match, create final list
    FINALFILELIST=`eval ${INCLUDEFINDCOMMAND} | sort`
else
    # Else create exclude find command

    EXCLUDEFINDCOMMAND=$(build_command "${EXCLUDEPATTERNS[@]}")

    # Compare the two lists, ignore the ones which are uniq to the exclude list, or appear in both lists.
    FINALFILELIST=`comm -23 <(eval ${INCLUDEFINDCOMMAND} | sort) <(eval ${EXCLUDEFINDCOMMAND} | sort)`
fi

# Print out all the included files.
for includedfile in ${FINALFILELIST[@]}; do
    echo "${includedfile}"
done

