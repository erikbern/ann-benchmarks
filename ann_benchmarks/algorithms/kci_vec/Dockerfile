FROM ann-benchmarks

# kci_vec: embedded vector search (C, single file). Installed from the evaluation wheel (binary, cosine only);
# it contains an AVX-512 and an AVX2 library and picks one by the CPU at import.
ARG KCI_VEC_WHEEL=https://github.com/KimYuki3/kci-vec-dist/releases/download/v0.1.1/kci_vec-0.1.1-py3-none-manylinux_2_35_x86_64.whl
RUN pip3 install "$KCI_VEC_WHEEL"
RUN python3 -c "import kci_vec, ctypes; p = kci_vec._find_lib(); ctypes.CDLL(p); print('kci_vec OK:', p)"
