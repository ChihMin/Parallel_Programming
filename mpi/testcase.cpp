#include <cstdio>
#include <cstdlib>

const int length = 1000000;
int arr[100000000];

int main(int argc, const char *argv[])
{
    int n = 84000000;
    FILE *f = fopen("testcase.in", "w");
    
    int now = 0;
    for (int i = 0; i < n; i++) {
        int sign = rand() % 2;
        int x = rand();
        if (sign) 
            x *= -1;
        
        arr[now++] = x; 
        if (now == length) {
            fwrite(arr, sizeof(int), length, f);  
            now = 0;
        }
    }
    fwrite(arr, sizeof(int), now, f);
    fclose(f);
    return 0;
}
