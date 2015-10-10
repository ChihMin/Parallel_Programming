#include <cstdio>
#include <cstdlib>

int main(int argc, const char *argv[])
{
    const int N = atoi(argv[1]);
    FILE *f = fopen(argv[2], "rb");
    
    int *x = new int[2147483647];
        
    for (int i = 0; i < N; i++) {
        int x;
        fread(&x, sizeof(int), 1, f);
        printf("Origin : %d\n", x);
    }
    delete [] x ;
    fclose(f);
    return 0;
}
