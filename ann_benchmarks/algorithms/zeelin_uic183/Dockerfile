FROM ann-benchmarks

RUN pip3 install --no-cache-dir numpy hnswlib faiss-cpu

RUN python3 -c 'import faiss, hnswlib; print(faiss.IndexFlatIP(4).d)'
