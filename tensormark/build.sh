#!/usr/bin/env bash
# Build the tensormark Python extension (and optionally the C++ test binaries).
#
#   ./build.sh            -> tm/tensormark.so   (importable via sys.path)
#   ./build.sh test_x.cpp -> build/test_x       (standalone gradcheck binary)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
PYINC=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PBINC=$($PY -c 'import pybind11; print(pybind11.get_include())')
# ACCELERATE_NEW_LAPACK: Apple deprecated the classic CBLAS interface in
# macOS 13.3; the updated one is ABI-compatible for our int-typed calls.
CXXFLAGS=${CXXFLAGS:-"-std=c++23 -O3 -march=native -DACCELERATE_NEW_LAPACK"}

if [ $# -eq 0 ]; then
    mkdir -p tm
    # Link to a temporary name and rename into place: replacing the file a
    # running interpreter has mapped would fault that process.
    tmp="tm/.tensormark.so.$$"
    # shellcheck disable=SC2086
    clang++ $CXXFLAGS -shared -undefined dynamic_lookup \
        -I"$PYINC" -I"$PBINC" \
        tensormark.cpp -o "$tmp" -framework Accelerate
    mv -f "$tmp" tm/tensormark.so
    echo "built tm/tensormark.so"
else
    mkdir -p build
    for src in "$@"; do
        out="build/$(basename "${src%.cpp}")"
        # shellcheck disable=SC2086
        clang++ $CXXFLAGS "$src" -o "$out" -framework Accelerate
        echo "built $out"
    done
fi
