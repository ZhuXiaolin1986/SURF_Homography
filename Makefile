TARGET = SURF_Homography

CFLAG = $(shell pkg-config --cflags opencv)
LFLAG = $(shell pkg-config --libs opencv)

$(TARGET):SURF_Homography.cpp
	g++ -g $(CFLAG)  SURF_Homography.cpp $(LFLAG) -o $(TARGET)

clean:
	rm -rfv $(TARGET)

