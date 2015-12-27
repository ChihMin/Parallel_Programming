#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <cstring>
#include <algorithm>
#define INF 1e9

int d[10001][10001];

int main(int argc, const char *argv[]){
 
  FILE *f = fopen("testcase.txt", "r");
  FILE *fout = fopen("output.txt", "w");

  int N, M;
  
  fscanf(f, "%d %d", &N, &M);
  for (int i = 1; i <=N; ++i) {
    for (int j = 1; j <= N; ++j) 
      d[i][j] = INF;
    d[i][i] = 0;  
  }

  for (int i = 0; i < M; i++) {
    int x , y, w;
    fscanf(f, "%d %d %d", &x, &y, &w);
    d[x][y] = w;
  }
  for (int k = 1; k <= N; ++k) {
    #pragma omp parallel num_threads(40) private(i)
    {
      int i;
      #pragma omp for schedule(static)
      for (i = 1; i <= N; ++i) {
        //printf("thread %d, i = %d\n", omp_get_thread_num(), i );
        for (int j = 1; j <= N; ++j)
          if (d[i][j] > d[i][k] + d[k][j])
            d[i][j] = d[i][k] + d[k][j];
      }
    }
  }

  for (int i = 1; i <= N; i++) {
    for (int j = 1; j < N ; ++j)
      fprintf(fout, "%d ", d[i][j]);
    fprintf(fout, "%d\n", d[i][N]);
  }

  fclose(fout);
  fclose(f);
  return 0;
}
