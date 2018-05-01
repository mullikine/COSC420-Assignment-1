#!/bin/bash
export TTY

force=n
while [ $# -gt 0 ]; do opt="$1"; case "$opt" in
    -f) {
        force=y
        shift
    }
    ;;

    *) break;
esac; done

pybin=python2

# set -xv

# The reason why I get the lack of terminal error is because this is
# being run from shanepy.


cd "$DUMP$HOME/notes2018/projects/nn-assignment-1/code/lab/"

# default test values
epochs=1000 # max epochs
regularization=0.0
momentum=0.5
learning_rate=0.1
decay=0


# TODO: softmax

# for ds in "3 Bit Parity" "4 Bit Parity" "Encoder" "Iris" "XOR"; do
# for test_type in regularization momentum learning_rate; do
# for a in sigmoid softmax; do
# for decay in 0.0001; do
# for ds in "XOR"; do
# for n_hiddens in 10; do
# for test_type in regularization momentum learning_rate; do

 # rectifier
for a in sigmoid softmax tanh; do
    for ds in "Iris"; do
        for test_type in regularization learning_rate; do
            for test_val in $(seq 0 0.1 0.5); do
                declare $test_type="$test_val"
                # eval ARG=\${$test_type}
                # echo "$ARG"
                # echo time,error
                dsdir="../datasets/$ds"
                dsdir="$(printf -- "%s" "$dsdir" | q -ftln)"
                # -H$n_hiddens  -m$momentum  -r$learning_rate
                cmd="$pybin ../perceptron.py -e$epochs -l$regularization -d$decay -a $a -t $dsdir"
                # echo "$cmd"
                # continue
                fn="results/$(pl "$test_type=$test_val ${epochs}i $ds $a" | mnm | slugify)"
                fdata="${fn}.time_vs_error.log"
                fout="${fn}.1.log"
                ferr="${fn}.2.log"
                if ! test "$force" = "y" && [ -f "$fout" ]; then
                    cat "$fout"
                else
                    # pl "$cmd" 1>&2
                    # continue
                    {
                        printf -- "%s\n" "$cmd -o $fdata" 1>&2
                        eval "$cmd -o $fdata"
                    } 1>"$fout" 2>"$ferr"
                    cat "$fout"
                fi
                # put it into a temporary file and make a graph. Figure out the
                # graph first.
            done
        done
    done
done
