CC=g++ -g
CFLAGS=-c -Wall
LDFLAGS=-pthread
SOURCES=RLPre.cpp main.cpp init.cpp test.cpp RL.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

