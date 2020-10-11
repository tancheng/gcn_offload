#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 20
#define I 2700
#define J 1400
#define K 2700


inline void reshape_2d(float* t_src, int t_I, int t_J, float** t_dest) {
  int i = 0;
  int j = 0;
  for (i=0; i<I; ++i) {
    for (j=0; j<J; ++j) {
      t_dest[i][j] = t_src[i*J+j];
    }
  }
}

inline void reshape_1d(float** t_src, int t_I, int t_J, float* t_dest) {
  int i = 0;
  int j = 0;
  int index = 0;
  for (i=0; i<I; ++i) {
    for (j=0; j<J; ++j) {
      t_dest[index] = t_src[i][j];
      ++index;
    }
  }
}

inline void reshape_1d_c2r(float* t_src, int t_I, int t_J, float* t_dest) {
  int i = 0;
  for (i = 0; i < t_I*t_J; ++i) {
    t_dest[(i%t_J)*t_I + i/t_J] = t_src[i];
  }
}

inline void reshape_2d_c2r(float** t_src, int t_I, int t_J, float** t_dest) {
  int i = 0;
  int j = 0;
  for (i = 0; i < t_I; ++i) {
    for (j = 0; j < t_J; ++j) {
      t_dest[i][j] = t_src[j][i];
    }
  }
}

void reshape_2d_tile(float** t_src, int t_I, int t_J, float**** t_dest) {
  int i = 0;
  int j = 0;
  int t = 0;
  int x = 0;
  int y = 0;
  for (x = 0; x < t_I/TILE; ++x) {
    for (y = 0; y < t_J/TILE; ++y) {
      for (i = 0; i < TILE; ++i) {
        for (j = 0; j < TILE; ++j) {
          t_dest[x][y][i][j] = t_src[x*TILE+i][y*TILE+j];
//          printf("[%d][%d][%d][%d]: %f ", x, y, i, j, t_dest[x][y][i][j]);
        }
//        printf("\n");
      }
//      printf("---------\n");
    }
//    printf("---------\n");
  }
}

void reshape_2d_tile_c2r(float** t_src, int t_I, int t_J, float**** t_dest) {
  int i = 0;
  int j = 0;
  int t = 0;
  int x = 0;
  int y = 0;
  for (x = 0; x < t_I/TILE; ++x) {
    for (y = 0; y < t_J/TILE; ++y) {
      for (i = 0; i < TILE; ++i) {
        for (j = 0; j < TILE; ++j) {
          t_dest[x][y][j][i] = t_src[x*TILE+i][y*TILE+j];
//          printf("[%d][%d][%d][%d]: %f ", x, y, i, j, t_dest[x][y][i][j]);
        }
//        printf("\n");
      }
//      printf("---------\n");
    }
//    printf("---------\n");
  }
}

void matmult_2d(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float** a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    a[i] = (float*)malloc(t_K * sizeof(float));

  float** b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    b[i] = (float*)malloc(t_J * sizeof(float));

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));


  reshape_2d(t_a, t_I, t_K, a);
  reshape_2d(t_b, t_K, t_J, b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
      float sub0 = 0;
      for (k = 0; k < t_K; k++) {
        sub0   = sub0 + a[i][k] * b[k][j];
      }
      c[i][j] = sub0;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float** a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    a[i] = (float*)malloc(t_K * sizeof(float));

  float** b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    b[i] = (float*)malloc(t_J * sizeof(float));

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));


  reshape_2d(t_a, t_I, t_K, a);
  reshape_2d(t_b, t_K, t_J, b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
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

      for (k = 0; k < t_K; k+=10) {
        sub0   = sub0 + a[i][k+0] * b[k+0][j];
        sub1   = sub1 + a[i][k+1] * b[k+1][j];
        sub2   = sub2 + a[i][k+2] * b[k+2][j];
        sub3   = sub3 + a[i][k+3] * b[k+3][j];
        sub4   = sub4 + a[i][k+4] * b[k+4][j];
        sub5   = sub5 + a[i][k+5] * b[k+5][j];
        sub6   = sub6 + a[i][k+6] * b[k+6][j];
        sub7   = sub7 + a[i][k+7] * b[k+7][j];
        sub8   = sub8 + a[i][k+8] * b[k+8][j];
        sub9   = sub9 + a[i][k+9] * b[k+9][j];
      }
      c[i][j] = sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_c2r(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float** a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    a[i] = (float*)malloc(t_K * sizeof(float));

  float** temp_b = (float**)malloc(t_J * sizeof(float*));
  for (i=0; i<t_J; ++i)
    temp_b[i] = (float*)malloc(t_K * sizeof(float));

  float** b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    b[i] = (float*)malloc(t_J * sizeof(float));

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));


  reshape_2d(t_a, t_I, t_K, a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_c2r(temp_b, t_K, t_J, b);

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
      float sub0 = 0;
      for (k = 0; k < t_K; k++) {
        sub0   = sub0 + a[i][k] * b[j][k];
      }
      c[i][j] = sub0;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_c2r_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float** a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    a[i] = (float*)malloc(t_K * sizeof(float));

  float** temp_b = (float**)malloc(t_J * sizeof(float*));
  for (i=0; i<t_J; ++i)
    temp_b[i] = (float*)malloc(t_K * sizeof(float));

  float** b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    b[i] = (float*)malloc(t_J * sizeof(float));

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));


  reshape_2d(t_a, t_I, t_K, a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_c2r(temp_b, t_K, t_J, b);

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
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

      for (k = 0; k < t_K; k+=10) {
        sub0   = sub0 + a[i][k+0] * b[j][k+0];
        sub1   = sub1 + a[i][k+1] * b[j][k+1];
        sub2   = sub2 + a[i][k+2] * b[j][k+2];
        sub3   = sub3 + a[i][k+3] * b[j][k+3];
        sub4   = sub4 + a[i][k+4] * b[j][k+4];
        sub5   = sub5 + a[i][k+5] * b[j][k+5];
        sub6   = sub6 + a[i][k+6] * b[j][k+6];
        sub7   = sub7 + a[i][k+7] * b[j][k+7];
        sub8   = sub8 + a[i][k+8] * b[j][k+8];
        sub9   = sub9 + a[i][k+9] * b[j][k+9];
      }
      c[i][j] = sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_1d_c2r(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float* b = malloc(t_K*t_J * sizeof(float));

  clock_t t;
  t = clock();

  reshape_1d_c2r(t_b, t_K, t_J, b);

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
      float sub0 = 0;
      for (k = 0; k < t_K; k++) {
        sub0   = sub0   + t_a[i*t_K + k + 0]  * b[j*t_K + k + 0];
      }
      t_c[i * t_J + j] = sub0;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);
  return;
}

void matmult_1d_c2r_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;

  float* b = malloc(t_K*t_J * sizeof(float));

  clock_t t;
  t = clock();

  reshape_1d_c2r(t_b, t_K, t_J, b);

  for (i = 0; i < t_I; i++) {
    for (j = 0; j < t_J; j++) {
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

      for (k = 0; k < t_K; k+=10) {
        sub0   = sub0   + t_a[i*t_K + k + 0]  * b[j*t_K + k + 0];
        sub1   = sub1   + t_a[i*t_K + k + 1]  * b[j*t_K + k + 1];
        sub2   = sub2   + t_a[i*t_K + k + 2]  * b[j*t_K + k + 2];
        sub3   = sub3   + t_a[i*t_K + k + 3]  * b[j*t_K + k + 3];
        sub4   = sub4   + t_a[i*t_K + k + 4]  * b[j*t_K + k + 4];
        sub5   = sub5   + t_a[i*t_K + k + 5]  * b[j*t_K + k + 5];
        sub6   = sub6   + t_a[i*t_K + k + 6]  * b[j*t_K + k + 6];
        sub7   = sub7   + t_a[i*t_K + k + 7]  * b[j*t_K + k + 7];
        sub8   = sub8   + t_a[i*t_K + k + 8]  * b[j*t_K + k + 8];
        sub9   = sub9   + t_a[i*t_K + k + 9]  * b[j*t_K + k + 9];
      }
      t_c[i * t_J + j] = sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);
  return;
}

void matmult_1d_tile_c2r(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int sj = 0;
  int i1 = 0;
  int j1 = 0;
  int k1 = 0;
  int mi, ki, kij;

  float* b = malloc(t_K*t_J * sizeof(float));

  clock_t t;
  t = clock();

  reshape_1d_c2r(t_b, t_K, t_J, b);

  for (i = 0; i < t_I; i+=TILE) {
    for (k = 0; k < t_K; k+=TILE) {
      for (j = 0; j < t_J; j+=TILE) {
        for (j1 = k; j1 < k+TILE; ++j1) {
          sj = t_K * j1;
          for (i1 = i; i1 < i+TILE; ++i1) {
            mi = t_K * i1;
            ki = t_J * i1;
            kij = ki + j1;
            float sub0 = 0;
            for (k1 = j; k1 < j+TILE; k1++) {
                sub0 += t_a[mi + k1 + 0] * b[sj + k1 + 0];
            }
            t_c[kij] += sub0;
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);
  return;
}

void matmult_1d_tile_c2r_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int sj = 0;
  int i1 = 0;
  int j1 = 0;
  int k1 = 0;
  int mi, ki, kij;

  float* b = malloc(t_K*t_J * sizeof(float));

  clock_t t;
  t = clock();

  reshape_1d_c2r(t_b, t_K, t_J, b);

  for (i = 0; i < t_I; i+=TILE) {
    for (k = 0; k < t_K; k+=TILE) {
      for (j = 0; j < t_J; j+=TILE) {
        for (j1 = k; j1 < k+TILE; ++j1) {
          sj = t_K * j1;

          for (i1 = i; i1 < i+TILE; ++i1) {
            mi = t_K * i1;
            ki = t_J * i1;
            kij = ki + j1;
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

            for (k1 = j; k1 < j+TILE; k1 += 10) {
                sub0 += t_a[mi + k1 + 0] * b[sj + k1 + 0];
                sub1 += t_a[mi + k1 + 1] * b[sj + k1 + 1];
                sub2 += t_a[mi + k1 + 2] * b[sj + k1 + 2];
                sub3 += t_a[mi + k1 + 3] * b[sj + k1 + 3];
                sub4 += t_a[mi + k1 + 4] * b[sj + k1 + 4];
                sub5 += t_a[mi + k1 + 5] * b[sj + k1 + 5];
                sub6 += t_a[mi + k1 + 6] * b[sj + k1 + 6];
                sub7 += t_a[mi + k1 + 7] * b[sj + k1 + 7];
                sub8 += t_a[mi + k1 + 8] * b[sj + k1 + 8];
                sub9 += t_a[mi + k1 + 9] * b[sj + k1 + 9];
            }
            t_c[kij] += sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);
  return;
}

void matmult_2d_tile(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int x = 0;
  int y = 0;

  int num_tile_a = t_I * t_K / (TILE * TILE);
  int num_tile_b = t_K * t_J / (TILE * TILE);

  float** temp_a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    temp_a[i] = (float*)malloc(t_K * sizeof(float));

  float**** a = (float****)malloc(t_I/TILE* sizeof(float***));
  for (x=0; x<t_I/TILE; ++x) {
    a[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      a[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        a[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** temp_b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    temp_b[i] = (float*)malloc(t_J * sizeof(float));

  float**** b = (float****)malloc(t_K/TILE* sizeof(float***));
  for (x=0; x<t_K/TILE; ++x) {
    b[x] = (float***)malloc(t_J/TILE * sizeof(float**));
    for (y=0; y<t_J/TILE; ++y) {
      b[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        b[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));

  reshape_2d(t_a, t_I, t_K, temp_a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_tile(temp_a, t_I, t_K, a);
  reshape_2d_tile(temp_b, t_K, t_J, b);

  int a_x = 0;
  int ab = 0;
  int b_y = 0;

  for (a_x = 0; a_x < t_I/TILE; a_x+=1) {
    for (ab = 0; ab < t_K/TILE; ab+=1) {
      for (b_y = 0; b_y < t_J/TILE; b_y+=1) {
        float sub0 = 0;
        for (i = 0; i < TILE; i++) {
          for (j = 0; j < TILE; ++j) {
            float sub0 = 0;
            for (k = 0; k < TILE; ++k) {
              sub0 = sub0 + a[a_x][ab][i][k] * b[ab][b_y][k][j];
//              printf("a[%d][%d][%d][%d] (%f) * b[%d][%d][%d][%d] (%f) = : %f\n", a_x, ab, i, k, a[a_x][ab][i][k], ab, b_y, k, j, b[ab][b_y][k][j], sub0);
            }
            c[a_x*TILE+i][b_y*TILE+j] += sub0;// + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
//            printf("c[%d][%d]: %f\n", a_x*TILE+i, b_y*TILE+j, c[a_x*TILE+i][b_y*TILE+j]);
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_tile_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int x = 0;
  int y = 0;

  int num_tile_a = t_I * t_K / (TILE * TILE);
  int num_tile_b = t_K * t_J / (TILE * TILE);

  float** temp_a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    temp_a[i] = (float*)malloc(t_K * sizeof(float));

  float**** a = (float****)malloc(t_I/TILE* sizeof(float***));
  for (x=0; x<t_I/TILE; ++x) {
    a[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      a[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        a[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** temp_b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    temp_b[i] = (float*)malloc(t_J * sizeof(float));

  float**** b = (float****)malloc(t_K/TILE* sizeof(float***));
  for (x=0; x<t_K/TILE; ++x) {
    b[x] = (float***)malloc(t_J/TILE * sizeof(float**));
    for (y=0; y<t_J/TILE; ++y) {
      b[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        b[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));

  reshape_2d(t_a, t_I, t_K, temp_a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_tile(temp_a, t_I, t_K, a);
  reshape_2d_tile(temp_b, t_K, t_J, b);

  int a_x = 0;
  int ab = 0;
  int b_y = 0;

  for (a_x = 0; a_x < t_I/TILE; a_x+=1) {
    for (b_y = 0; b_y < t_J/TILE; b_y+=1) {
      for (ab = 0; ab < t_K/TILE; ab+=1) {
   
        for (i = 0; i < TILE; i++) {
          for (j = 0; j < TILE; ++j) {
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
 
            for (k = 0; k < TILE; k+=10) {
              sub0 = sub0 + a[a_x][ab][i][k+0] * b[ab][b_y][k+0][j];
              sub1 = sub1 + a[a_x][ab][i][k+1] * b[ab][b_y][k+1][j];
              sub2 = sub2 + a[a_x][ab][i][k+2] * b[ab][b_y][k+2][j];
              sub3 = sub3 + a[a_x][ab][i][k+3] * b[ab][b_y][k+3][j];
              sub4 = sub4 + a[a_x][ab][i][k+4] * b[ab][b_y][k+4][j];
              sub5 = sub5 + a[a_x][ab][i][k+5] * b[ab][b_y][k+5][j];
              sub6 = sub6 + a[a_x][ab][i][k+6] * b[ab][b_y][k+6][j];
              sub7 = sub7 + a[a_x][ab][i][k+7] * b[ab][b_y][k+7][j];
              sub8 = sub8 + a[a_x][ab][i][k+8] * b[ab][b_y][k+8][j];
              sub9 = sub9 + a[a_x][ab][i][k+9] * b[ab][b_y][k+9][j];
//              printf("a[%d][%d][%d][%d] (%f) * b[%d][%d][%d][%d] (%f) = : %f\n", a_x, ab, i, k, a[a_x][ab][i][k], ab, b_y, k, j, b[ab][b_y][k][j], sub0);
            }
            c[a_x*TILE+i][b_y*TILE+j] += sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
//            printf("c[%d][%d]: %f\n", a_x*TILE+i, b_y*TILE+j, c[a_x*TILE+i][b_y*TILE+j]);
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_tile_c2r(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int x = 0;
  int y = 0;

  int num_tile_a = t_I * t_K / (TILE * TILE);
  int num_tile_b = t_K * t_J / (TILE * TILE);

  float** temp_a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    temp_a[i] = (float*)malloc(t_K * sizeof(float));

  float**** a = (float****)malloc(t_I/TILE* sizeof(float***));
  for (x=0; x<t_I/TILE; ++x) {
    a[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      a[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        a[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** temp_b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    temp_b[i] = (float*)malloc(t_J * sizeof(float));

  float**** b = (float****)malloc(t_J/TILE* sizeof(float***));
  for (x=0; x<t_J/TILE; ++x) {
    b[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      b[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        b[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));

  reshape_2d(t_a, t_I, t_K, temp_a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_tile(temp_a, t_I, t_K, a);
  reshape_2d_tile_c2r(temp_b, t_K, t_J, b);

  int a_x = 0;
  int ab = 0;
  int b_y = 0;

  for (a_x = 0; a_x < I/TILE; a_x+=1) {
    for (b_y = 0; b_y < J/TILE; b_y+=1) {
      for (ab = 0; ab < K/TILE; ab+=1) {
   
        for (i = 0; i < TILE; i++) {
          for (j = 0; j < TILE; ++j) {
            float sub0 = 0;

            for (k = 0; k < TILE; k++) {
              sub0 = sub0 + a[a_x][ab][i][k] * b[ab][b_y][j][k];
//              printf("a[%d][%d][%d][%d] (%f) * b[%d][%d][%d][%d] (%f) = : %f\n", a_x, ab, i, k, a[a_x][ab][i][k], ab, b_y, k, j, b[ab][b_y][k][j], sub0);
            }
            c[a_x*TILE+i][b_y*TILE+j] += sub0;
//            printf("c[%d][%d]: %f\n", a_x*TILE+i, b_y*TILE+j, c[a_x*TILE+i][b_y*TILE+j]);
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}

void matmult_2d_tile_c2r_vec(float* t_a, float* t_b, float* t_c, int t_I, int t_J, int t_K) {
  int i = 0;
  int j = 0;
  int k = 0;
  int x = 0;
  int y = 0;

  int num_tile_a = t_I * t_K / (TILE * TILE);
  int num_tile_b = t_K * t_J / (TILE * TILE);

  float** temp_a = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    temp_a[i] = (float*)malloc(t_K * sizeof(float));

  float**** a = (float****)malloc(t_I/TILE* sizeof(float***));
  for (x=0; x<t_I/TILE; ++x) {
    a[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      a[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        a[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** temp_b = (float**)malloc(t_K * sizeof(float*));
  for (i=0; i<t_K; ++i)
    temp_b[i] = (float*)malloc(t_J * sizeof(float));

  float**** b = (float****)malloc(t_J/TILE* sizeof(float***));
  for (x=0; x<t_J/TILE; ++x) {
    b[x] = (float***)malloc(t_K/TILE * sizeof(float**));
    for (y=0; y<t_K/TILE; ++y) {
      b[x][y] = (float**)malloc(TILE * sizeof(float*));
      for (i=0; i<TILE; ++i) {
        b[x][y][i] = (float*)malloc(TILE * sizeof(float));
      }
    }
  }

  float** c = (float**)malloc(t_I * sizeof(float*));
  for (i=0; i<t_I; ++i)
    c[i] = (float*)malloc(t_J * sizeof(float));

  reshape_2d(t_a, t_I, t_K, temp_a);
  reshape_2d(t_b, t_K, t_J, temp_b);
  reshape_2d(t_c, t_I, t_J, c);

  /*float* c = malloc(nay * sizeof(float));*/
  clock_t t; 
  t = clock();

  reshape_2d_tile(temp_a, t_I, t_K, a);
  reshape_2d_tile_c2r(temp_b, t_K, t_J, b);

  int a_x = 0;
  int ab = 0;
  int b_y = 0;

  for (a_x = 0; a_x < I/TILE; a_x+=1) {
    for (b_y = 0; b_y < J/TILE; b_y+=1) {
      for (ab = 0; ab < K/TILE; ab+=1) {
   
        for (i = 0; i < TILE; i++) {
          for (j = 0; j < TILE; ++j) {
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
 
            for (k = 0; k < TILE; k+=10) {
              sub0 = sub0 + a[a_x][ab][i][k+0] * b[ab][b_y][j][k+0];
              sub1 = sub1 + a[a_x][ab][i][k+1] * b[ab][b_y][j][k+1];
              sub2 = sub2 + a[a_x][ab][i][k+2] * b[ab][b_y][j][k+2];
              sub3 = sub3 + a[a_x][ab][i][k+3] * b[ab][b_y][j][k+3];
              sub4 = sub4 + a[a_x][ab][i][k+4] * b[ab][b_y][j][k+4];
              sub5 = sub5 + a[a_x][ab][i][k+5] * b[ab][b_y][j][k+5];
              sub6 = sub6 + a[a_x][ab][i][k+6] * b[ab][b_y][j][k+6];
              sub7 = sub7 + a[a_x][ab][i][k+7] * b[ab][b_y][j][k+7];
              sub8 = sub8 + a[a_x][ab][i][k+8] * b[ab][b_y][j][k+8];
              sub9 = sub9 + a[a_x][ab][i][k+9] * b[ab][b_y][j][k+9];
//              printf("a[%d][%d][%d][%d] (%f) * b[%d][%d][%d][%d] (%f) = : %f\n", a_x, ab, i, k, a[a_x][ab][i][k], ab, b_y, k, j, b[ab][b_y][k][j], sub0);
            }
            c[a_x*TILE+i][b_y*TILE+j] += sub0 + sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9;
//            printf("c[%d][%d]: %f\n", a_x*TILE+i, b_y*TILE+j, c[a_x*TILE+i][b_y*TILE+j]);
          }
        }
      }
    }
  }

  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
  printf("*** [custom C inside lib] mm: %f seconds ***\n", time_taken);

  reshape_1d(c, t_I, t_J, t_c);

  return;
}
