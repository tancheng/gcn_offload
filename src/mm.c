#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define xtile 2
#define ytile 4
#define ztile 800

void matmult(float* a, float* b, float* c, int I, int J, int K) {
    int x = 0;
    int y = 0;
    int z = 0;
    int i = 0;
    int j = 0;
    int k = 0;

    /*float* c = malloc(nay * sizeof(float));*/
    clock_t t; 
    t = clock();

    //TODO: should consider tiling again, since now it does not work well.
    for (x = 0; x < I; x+=xtile) {
      for (i = x; i < x+xtile; i++) {
//        for (j = 0; j < J; j++) {
        for (y = 0; y < J; y+=ytile) {
          for (j = y; j < y+ytile; j++) {
            float sub0 = 0;
            float sub1 = 0;
            float sub2 = 0;
            float sub3 = 0;

            float sub4 = 0;
            float sub5 = 0;
            float sub6 = 0;
            float sub7 = 0;

            float sub8 = 0;
            float sub9 = 0;
//            float sub10 = 0;
//            float sub11 = 0;
//
//            float sub12 = 0;
//            float sub13 = 0;
//            float sub14 = 0;
//            float sub15 = 0;
//
//            float sub16 = 0;
//            float sub17 = 0;
//            float sub18 = 0;
//            float sub19 = 0;

            for (z = 0; z < K; z+=ztile) {
              for (k = z; k < z+ztile; k+=10) {
//                sub = sub + a[i * K + k] * b[k * J + j];
                sub0   = sub0   + a[i*K + k + 0]  * b[j*K + k + 0];
                sub1   = sub1   + a[i*K + k + 1]  * b[j*K + k + 1];
                sub2   = sub2   + a[i*K + k + 2]  * b[j*K + k + 2];
                sub3   = sub3   + a[i*K + k + 3]  * b[j*K + k + 3];
                sub4   = sub4   + a[i*K + k + 4]  * b[j*K + k + 4];
                sub5   = sub5   + a[i*K + k + 5]  * b[j*K + k + 5];
                sub6   = sub6   + a[i*K + k + 6]  * b[j*K + k + 6];
                sub7   = sub7   + a[i*K + k + 7]  * b[j*K + k + 7];
                sub8   = sub8   + a[i*K + k + 8]  * b[j*K + k + 8];
                sub9   = sub9   + a[i*K + k + 9]  * b[j*K + k + 9];

//                sub10  = sub10  + a[temp_i + k + 10] * b[temp_j   + k + 10];
//                sub11  = sub11  + a[temp_i + k + 11] * b[temp_j   + k + 11];
//                sub12  = sub12  + a[temp_i + k + 12] * b[temp_j   + k + 12];
//                sub13  = sub13  + a[temp_i + k + 13] * b[temp_j   + k + 13];
//                sub14  = sub14  + a[temp_i + k + 14] * b[temp_j   + k + 14];
//                sub15  = sub15  + a[temp_i + k + 15] * b[temp_j   + k + 15];
//
//                sub16  = sub16  + a[temp_i + k + 16] * b[temp_j   + k + 16];
//                sub17  = sub17  + a[temp_i + k + 17] * b[temp_j   + k + 17];
//                sub18  = sub18  + a[temp_i + k + 18] * b[temp_j   + k + 18];
//                sub19  = sub19  + a[temp_i + k + 19] * b[temp_j   + k + 19];

//                sub8  = sub8  + a[temp_i + k + 0] * b[temp_jjj  + k + 0];
//                sub9  = sub9  + a[temp_i + k + 1] * b[temp_jjj  + k + 1];
//                sub10 = sub10 + a[temp_i + k + 2] * b[temp_jjj  + k + 2];
//                sub11 = sub11 + a[temp_i + k + 3] * b[temp_jjj  + k + 3];
//
//                sub12 = sub12 + a[temp_i + k + 0] * b[temp_jjjj + k + 0];
//                sub13 = sub13 + a[temp_i + k + 1] * b[temp_jjjj + k + 1];
//                sub14 = sub14 + a[temp_i + k + 2] * b[temp_jjjj + k + 2];
//                sub15 = sub15 + a[temp_i + k + 3] * b[temp_jjjj + k + 3];

              }
            }
//            float temp_sum0 = sub0  + sub1  + sub2  + sub3  + sub4;
//            float temp_sum1 = sub5  + sub6  + sub7;//  + sub8  + sub9;
//            float temp_sum2 = sub10 + sub11 + sub12 + sub13 + sub14;
//            float temp_sum3 = sub15 + sub16 + sub17 + sub18 + sub19;
            c[i * J + j + 0] = sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;

//            c[i * J + j + 0] = sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7;
//            c[i * J + j + 0] += sub8  + sub9  + sub10 + sub11 + sub12 + sub13 + sub14 + sub15;
//            c[i * J + j + 0] += sub16  + sub17  + sub18 + sub19;

//            printf("c[%d][%d]: %f", i, j, sub);
          }
        }
      }
    }

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
    printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);
    return;
}
