CXX = hipcc
CXX_FLAGS = -std=c++17 -lX11 -lpthread

PROGNAME = smoke

.PHONY: clean all

all: $(PROGNAME)

clean:
	rm $(PROGNAME)
	rm -rf out/

$(PROGNAME): main.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@
	mkdir -p out
