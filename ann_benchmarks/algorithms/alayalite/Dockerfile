FROM ann-benchmarks

RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/AlayaDB-AI/AlayaLite/releases/download/v0.1.0a2/alayalite-0.1.0a2-cp310-cp310-linux_x86_64.whl -O /home/app/alayalite-0.1.0a2-cp310-cp310-linux_x86_64.whl
RUN python3 -m pip install /home/app/alayalite-0.1.0a2-cp310-cp310-linux_x86_64.whl
RUN python3 -c 'import alayalite'