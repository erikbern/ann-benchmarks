FROM ann-benchmarks

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    mkdir -p /usr/local/lib && \
    curl -fsSL -o /usr/local/lib/libsilicium_ann_x86_64.so https://github.com/silicium369-spec/silicium-ann/releases/download/ann-v1.0.0/libsilicium_ann_x86_64.so && \
    echo "59e92c84a62667e83cde0f64bdb62bcbb91036898ba023895ba0d9d3c60e3d60  /usr/local/lib/libsilicium_ann_x86_64.so" | sha256sum -c - && \
    ldconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -c 'import ctypes; ctypes.CDLL("/usr/local/lib/libsilicium_ann_x86_64.so"); print("[silicium] Build environment verified")'
