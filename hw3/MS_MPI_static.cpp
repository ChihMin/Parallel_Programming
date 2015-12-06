 /* 
   Sequential Mandelbort sort
 */

#include "mpi.h"
#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <cstring>

#define atof(tar, index) sscanf(argv[index], "%lf", &tar)

typedef struct complextype
{
	double real, imag;
} Compl;

double minX, minY;
double maxX, maxY;
int threads;
int width = 800, height = 800;
bool isEnable = false;


int main(int argc, char **argv)
{
    threads = atoi(argv[1]);
    
    atof(minX, 2);
    atof(maxX, 3);
    atof(minY, 4);
    atof(maxY, 5);
    width = atoi(argv[6]);
    height = atoi(argv[7]);
    
    std::string in = argv[8];
    if (in == "enable")
        isEnable = true;
    else
        isEnable = false;
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
     
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
    
	/* draw points */
    int numPerTask = width / size;
    if (rank == size - 1)
       numPerTask += width % size;
    
    int tmp = 0;
    if (rank == 0) tmp = -1;
    else    tmp = 123; 
     
    printf("rank %d -> %p\n", rank, display);
    
    if (rank == 0) {
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
        if (isEnable)  
             XFlush(display);
    }
      
    MPI_Bcast(&display, sizeof(Display*), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gc, sizeof(GC), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&window, sizeof(Window), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tmp, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD);
    
    printf("rank %d -> %p\n", rank, display);

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = numPerTask * rank, k = 0; k < numPerTask; i++, k++) {
        //printf("[OUT] thread = %d, i = %d\n", omp_get_thread_num(), i);	
        for(int j=0; j<height; j++) {
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
            if (rank == 1) {   
                sleep(5);
                XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
                XDrawPoint (display, window, gc, i, j);
            }
        }
    }

    MPI_Finalize();
	return 0;
}
