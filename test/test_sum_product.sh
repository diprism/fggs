#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

round () {
    # Round all floats to 1 decimal place.
    perl -pe 's/(-?\d+(\.\d+(e-?\d+)?)?)/sprintf("%.1f",$1)/ge; s/ //g;'
}

find_file_ans() {
    local dir=${1}
    local a=$(grep -ir "correct" "${dir}" | sed 's/:-- correct: /\@/g' | sed 's/ //g')
    echo ${a}
}

test_cases_good=$(find_file_ans "tests/good")
all_test_cases="${test_cases_good}"
all_test_cases=(${all_test_cases// / })
case_number=${#all_test_cases[@]}
error_number=0

for case in "${all_test_cases[@]}"; do
    file_and_ans=(${case//@/ })
    file=${file_and_ans[0]}
    ans=${file_and_ans[1]}
    printf '%-40s' "Testing ${file}... "
    result=$(./perplc $file | ${PYTHON:-python} bin/sum_product.py -g -d /dev/stdin)
    if [ $? = 0 ]; then
        correct=$(echo ${ans} | round)
        out=$(echo ${result} | round)
        if [ -n "$correct" ]; then
            if [ "$correct" = "$out" ]; then
                echo "Success"
            else
                echo "Failure"
                echo "Incorrect result in ${file}: ${correct} != ${out}"
                error_number=$((error_number+1))
            fi
        fi
    else
        echo "Failure"
        echo "Error in ${file}: the program returned a non-zero code"
        error_number=$((error_number+1))
    fi
done

if [ "${error_number}" -eq "0" ]; then
    echo "${case_number}/${case_number} test(s) passed!"
    exit 0
else
    echo "${error_number}/${case_number} test(s) failed"
    exit 1
fi
