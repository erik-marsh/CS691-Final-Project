CXX = g++.exe
CXX_FLAGS = -std=c++17

PROGNAME = smoke.exe

.PHONY: clean all

all: $(PROGNAME)

clean:
	powershell.exe "Remove-Item $(PROGNAME); $$null"

$(PROGNAME): main.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@