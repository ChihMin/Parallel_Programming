 /* 
   Sequential Mandelbort sort
 */

#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>
#include <string>
#include <cstring>

#define LOCK(mutex) pthread_mutex_lock(&mutex)
#define UNLOCK(mutex) pthread_mutex_unlock(&mutex)
#define atof(tar, index) sscanf(argv[index], "%lf", &tar)

typedef struct complextype
{
	double real, imag;
} Compl;

pthread_mutex_t drawMutex;
double minX, minY;
double maxX, maxY;
int threads;
int width = 800, height = 800;
bool isEnable = false;

using namespace std;

int main(int argc, char **argv)
{
    threads = atoi(argv[1]);
    atof(minX, 2);
    atof(maxX, 3);
    atof(minY, 4);
    atof(maxY, 5);
    width = atoi(argv[6]);
    height = atoi(argv[7]);
    
    string in = argv[8];
    if (in == "enable")
        isEnable = true;
    else
        isEnable = false;
    
    
    printf("threads = %d\n", threads); 
     
    Display *display;
	Window window;      //initialization for a window
	int screen;         //which screen 

	/* set window position */
	int x = 0;
	int y = 0;
	
    /* create graph */
	GC gc;
	XGCValues values;
	long valuemask = 0;

	/* border width in pixels */
	int border_width = 0;
	/* open connection with the server */
    
    if (!isEnable)
        goto MAIN_LOOP;
         
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return 0;
	}

	screen = DefaultScreen(display);

	/* set window size */


	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
					BlackPixel(display, screen), WhitePixel(display, screen));
	
	
	gc = XCreateGC(display, window, valuemask, &values);
	//XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
    
    
MAIN_LOOP:  	
    pthread_mutex_init(&drawMutex, NULL);

	/* draw points */
	int i, j;
    int points[2000];
    memset(points, 0, sizeof(points));
    #pragma omp parallel num_threads(threads) private(i, j)
    {
        double start = omp_get_wtime();
        #pragma omp for schedule(dynamic, 1) nowait
        for(i=0; i<width; i++) {
            //printf("[OUT] thread = %d, i = %d\n", omp_get_thread_num(), i);	
            points[omp_get_thread_num()]++;
            for(j=0; j<height; j++) {
                Compl z, c;
                double temp, lengthsq;
                int repeats;
                z.real = 0.0;
                z.imag = 0.0;
                
                double scaleX = width / (maxX - minX);
                double scaleY = height / (maxY - minY);
                
                c.real = ((double)i + scaleX * minX) / scaleX; /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
                c.imag = ((double)j + scaleY * minY) / scaleY;; /* So needs to scale the window */
                repeats = 0;
                lengthsq = 0.0;

                while(repeats < 10000 && lengthsq < 4.0) { /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
                    temp = z.real*z.real - z.imag*z.imag + c.real;
                    z.imag = 2*z.real*z.imag + c.imag;
                    z.real = temp;
                    lengthsq = z.real*z.real + z.imag*z.imag; 
                    repeats++;
                }
                
                if (!isEnable) continue;    
                LOCK(drawMutex);
                    XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
                    XDrawPoint (display, window, gc, i, j);
                UNLOCK(drawMutex);
            }
        }
        double end = omp_get_wtime();
        printf("TIMER: rank %d time %.6lf points %d\n", omp_get_thread_num(), end - start, points[omp_get_thread_num()]);
    }

    if (isEnable)  
         XFlush(display);
	// sleep(1000);
	return 0;
}
