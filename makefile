
FLAGS = 
FLAGS += -Wall
FLAGS += -fopenmp
FLAGS += -lm

all: logreg_breast_cancer

logreg_breast_cancer: logreg.o matrix.o
	gcc obj/*.o examples/logreg_breast_cancer.c $(FLAGS)  -o bin/logreg_breast_cancer	

logreg.o:  
	gcc -c src/logreg.c $(FLAGS) -o obj/logreg.o

matrix.o:
	gcc -c src/matrix.c $(FLAGS) -o obj/matrix.o
