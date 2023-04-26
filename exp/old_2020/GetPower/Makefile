EXEC := appPower
LIBS := -Wall -Wextra -O2
CC   := gcc

all: $(EXEC)

$(EXEC): appPower.o
	$(CC) $^ -o $@ $(LIBS)
	sudo cp appPower getPower /usr/local/bin/

clean:
	rm -f *.o $(EXEC)
