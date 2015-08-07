
all : main

main : main.o lmsvd.o
	g++ -o main main.o lmsvd.o -lblas -llapack

main.o : main.cpp common.h lmsvd.h
	g++ -c main.cpp

lmsvd.o : lmsvd.cpp common.h lmsvd.h
	g++ -c lmsvd.cpp -std=c++11

