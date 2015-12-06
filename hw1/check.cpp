#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <algorithm>

#define ROOT 0

int main(int argc, char *argv[])
{
    FILE *f = fopen(argv[1], "r");
    int x;
    while(fread(&x, sizeof(int), 1, f))
        printf("%d\n", x);
    return 0;
}
