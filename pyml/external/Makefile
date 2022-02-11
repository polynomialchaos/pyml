TARGET = libexternal
CFLAGS = -O3 -fopenmp

.PHONY: default all clean

default: $(TARGET)
all: default

$(TARGET):
	gcc -shared $(CFLAGS) -o $(TARGET).so cross_correlate_2d.c

clean:
	rm -f $(TARGET).so