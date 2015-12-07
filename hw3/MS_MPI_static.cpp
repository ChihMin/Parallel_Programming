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
#define ROOT 0
#define send(buf, count, type, dest) \
  MPI_Send(buf, count, type, dest, 0, MPI_COMM_WORLD);

#define recv(buf, count, type, source) \
  MPI_Recv(buf, count, type, source, \
              MPI_ANY_TAG, MPI_COMM_WORLD, &status);

typedef struct complextype
{
	double real, imag;
} Compl;

struct Pixel {
    int i, j, repeats;
    Pixel(){};
    Pixel(int _i, int _j, int _repeat) {
        i = _i, j = _j, repeats = _repeat;
    }
};

double minX, minY;
double maxX, maxY;
int threads;
int width = 800, height = 800;
bool isEnable = false;
bool isCheck[800][800];

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

/********* MPI MISSION START *********/
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    MPI_Status status;
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
    int numPerTask = width / (size);
    if (rank == size - 1)
       numPerTask += width % (size);
    
    if (rank == 0) {
        display = XOpenDisplay(NULL);
        if(display == NULL) {
            fprintf(stderr, "cannot open display\n");
            return 0;
        }

        screen = DefaultScreen(display);

        /* set window size */
        
        /* create window */
        window = XCreateSimpleWindow(display, 
            RootWindow(display, screen), x, y, 
                width, height, border_width, 
                    BlackPixel(display, screen), 
                        WhitePixel(display, screen));
        
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

    Pixel *pixel = new Pixel[numPerTask * height];
    int curIndex = 0;
    int beginPos = rank != size - 1 ?  
        numPerTask * (rank) : (rank) * (numPerTask - (width % (size)));
    
     
    printf("rank %d -> begin = %d, numTasks = %d\n", rank, beginPos, numPerTask);
    for(int i = beginPos, k = 0; k < numPerTask; i++, k++) {
        for(int j=0; j<height; j++) {
            // printf("rank %d : (%d, %d)\n", rank, i, j);
            Compl z, c;
            double temp, lengthsq;
            int repeats;
            z.real = 0.0;
            z.imag = 0.0;
            
            double scaleX = width / (maxX - minX);
            double scaleY = height / (maxY - minY);
            
            c.real = ((double)i + scaleX * minX) / scaleX; 
            /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
            
            c.imag = ((double)j + scaleY * minY) / scaleY; 
            /* So needs to scale the window */

            repeats = 0;
            lengthsq = 0.0;

            while(repeats < 10000 && lengthsq < 4.0) { 
                /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real + z.imag*z.imag;
                 
                repeats++;
            }
            
            if (!isEnable) continue; 
            pixel[curIndex++] = Pixel(i, j, repeats);
        }
    }
    if (rank != 0) {
        send(&curIndex, 1, MPI_INT, ROOT);
       // printf("rank %d send curIndex %d ...\n", rank, curIndex);
       // sleep(5);
        send(pixel, curIndex * sizeof(Pixel), MPI_CHAR, ROOT);
       // printf("rank %d send pixel...\n", rank);
    }
    else {
        memset(isCheck, 0, sizeof(isCheck));
        for (int index = 0; index < curIndex; ++index) {
            isCheck[pixel[index].i][pixel[index].j] = 1;
            XSetForeground (display, gc,  
                        1024 * 1024 * (pixel[index].repeats % 256));	
            XDrawPoint (display, window, gc, pixel[index].i, pixel[index].j);
        }
            
         
        for (int threads = 1; threads < size; ++threads) {
            int pixelNum;
            recv(&pixelNum, sizeof(int), MPI_INT, threads);
            // printf("recv pixelNum =  %d\n", pixelNum);   
            Pixel *pixel = new Pixel[pixelNum];
            recv(pixel, pixelNum * sizeof(Pixel), MPI_CHAR, threads);
            for (int j = 0; j < pixelNum; ++j) {
                isCheck[pixel[j].i][pixel[j].j] = 1;
                XSetForeground (display, gc,  
                            1024 * 1024 * (pixel[j].repeats % 256));	
                XDrawPoint (display, window, gc, pixel[j].i, pixel[j].j);
            }
            delete [] pixel;
        }
    
        for (int i = 0; i < width; ++i)
            for (int j = 0; j < height; ++j)
                if (!isCheck[i][j])
                    printf("check : %d %d\n", i, j);
    }
    delete [] pixel;

    MPI_Finalize();
    sleep(100);
	return 0;
}
