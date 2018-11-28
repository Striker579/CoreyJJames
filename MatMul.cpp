#include <omp.h>
#include "MatMul.h"

#define version 1
// matmul returns the product c = a * b
//

#if version == 1
void matmul(const double a[][N], const double b[][N], double c[][N], int n) {
#pragma omp parallel for 
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

#elif version == 2
void matmul(const double a[][N], const double b[][N], double c[][N], int n) {
	int i, j, k;
#pragma omp parallel for schedule(dynamic) private(i, j, k) shared(a,b,n)
	for (i = 0; i < n; i++) {
		for (k = 0; k < n; k++) {
			for (j = 0; j < n; j++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}
#endif
// checksum returns the sum of the coefficients in matrix x[N][N]
//
double checksum(const double x[][N]) {
	double sum = 0.0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			sum += x[i][j];
		}
	}
	return sum;
	
}

// initialize initializes matrix a[N][N]
//
void initialize(double a[][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = static_cast<double>(i * j) / (N * N);
		}
	}
}