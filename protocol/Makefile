override LDFLAGS += -pthread

all: fr-dolphinn

fr-dolphinn: Dolphinn frontend.o dolphinn.o
	g++ -o fr-dolphinn frontend.o dolphinn.o $(LDFLAGS)

Dolphinn:
	git clone https://github.com/gsamaras/Dolphinn || true

frontend.o: ../install/ann-filters/wrappers/frontend/frontend.c
	gcc -c -o $@ $(CFLAGS) $^

clean:
	rm -f frontend.o dolphinn.o
