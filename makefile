
FLAGS = 
FLAGS += -Wall
FLAGS += -fopenmp
FLAGS += -lm
FLAGS += -fPIC 

all: logreg_breast_cancer

logreg_breast_cancer: logreg.o matrix.o
	gcc obj/*.o examples/logreg_breast_cancer.c $(FLAGS)  -o bin/logreg_breast_cancer	

libs:  matrix.so logreg.so

logreg.so: logreg.o matrix.o
	gcc -shared obj/logreg.o obj/matrix.o $(FLAGS) -o bin/logreg.so

matrix.so: matrix.o
	gcc -shared obj/matrix.o $(FLAGS) -o bin/matrix.so

logreg.o:  
	gcc -c src/logreg.c $(FLAGS) -o obj/logreg.o

matrix.o:
	gcc -c src/matrix.c $(FLAGS) -o obj/matrix.o
