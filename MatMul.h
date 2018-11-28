#pragma once
#define N 2048
typedef double array[N];

void initialize(double a[][N]);
void matmul(const double a[][N], const double b[][N], double c[][N], int n);
double checksum(const double x[][N]);