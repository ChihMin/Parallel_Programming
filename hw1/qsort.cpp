#include <cstdio>
#include <algorithm>

using namespace std;

int main(int argc, const char *argv[])
{
    int n = atoi(argv[1]);
    FILE *f = fopen(argv[2], "r");
    
    int *arr  = new int[n+1];
    fread(arr, sizeof(int), n, f);

    sort(arr, arr+n);
    fwrite(arr, sizeof(int), n, f);

    return 0;
}
