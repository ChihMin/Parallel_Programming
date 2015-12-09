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
#include <vector>

#define atof(tar, index) sscanf(argv[index], "%lf", &tar)
#define ROOT 0
#define send(buf, count, type, dest) \
  MPI_Send(buf, count, type, dest, 0, MPI_COMM_WORLD);

#define recv(buf, count, type, source) \
  MPI_Recv(buf, count, type, source, \
              MPI_ANY_TAG, MPI_COMM_WORLD, &status);

#define Isend(buf, count, type, dest, req) \
  MPI_Isend(buf, count, type, dest, 0, MPI_COMM_WORLD, req);

#define Irecv(buf, count, type, src, req) \
  MPI_Irecv(buf, count, type, src, MPI_ANY_TAG, \
                MPI_COMM_WORLD, req); 


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
    double start, end;
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    start = MPI_Wtime();

    MPI_Status status;
    MPI_Request request[size];
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
    int points = 0;
    if (rank == size - 1)
       numPerTask += width % (size);
    
    if (isEnable && rank == 0) {
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
    
    MPI_Barrier(MPI_COMM_WORLD); 
    if (rank == 0 && size != 1) {
        bool *isReady = new bool[size];
        int thread = 1;
        bool isLaunch; 
        bool gar; 
        for (int i = 1; i < size; ++i)
            Irecv(&gar, 1, MPI_CHAR, i, &request[i]); 

        for (int i = 0; i < width; ++i) {
            isLaunch = false;
            while (!isLaunch) {
                int flag;
                MPI_Test(&request[thread], &flag, &status);
                if (flag) {
                    Irecv(&gar, 1, MPI_CHAR, thread, &request[thread]); 
                    send(&i, 1, MPI_INT, thread);
                    isLaunch = true;
                } 
                /* Round robin ... */
                thread = (thread + 1) % size;
                if (thread == 0)
                    thread = (thread + 1 ) % size;
            }
        }
        
        for (int i = 1; i < size; ++i) {
            int flag;
            do {
                MPI_Test(&request[i], &flag, &status);
            } while(!flag);
            int stop = -1;
            send(&stop, 1, MPI_INT, i);
        }
    }
    else {
        std::vector<Pixel> pixelArray;
        int curIndex = 0;
        int beginPos = rank != size - 1 ?  
            numPerTask * (rank) : (rank) * (numPerTask - (width % (size)));
        
         
        /*** Tell ROOT I'm ready ***/
        bool isLaunch = true;
        int i = width - 1;
        if (rank != 0) {
          Isend(&isLaunch, 1, MPI_CHAR, ROOT, &request[rank]); 
         
          recv(&i, 1, MPI_INT, ROOT);
        }
        while (i != -1) {
            points++;
            for (int j = 0; j < height; j++) {
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
                pixelArray.push_back(Pixel(i, j, repeats));
            }
            if (size == 1)  i--;
            if (rank != ROOT) {
              Isend(&isLaunch, 1, MPI_CHAR, ROOT, &request[rank]); 
              recv(&i, 1, MPI_INT, ROOT);
            }
        }

        if (size == 1 && isEnable) {
            int drawTimes = 2;
            while (drawTimes--)
            for (int k = 0; k < pixelArray.size(); ++k) {
              int i = pixelArray[k].i;
              int j = pixelArray[k].j;
              int repeats = pixelArray[k].repeats;
              XSetForeground (display, gc,  
                        1024 * 1024 * (repeats % 256));	
              XDrawPoint (display, window, gc, i, j);
            }
        }
        else if (isEnable) { 
            Pixel *pixel = new Pixel[pixelArray.size()];
            for (int i = 0; i < pixelArray.size(); ++i)
                pixel[i] = pixelArray[i];
            curIndex = pixelArray.size();
            send(&curIndex, 1, MPI_INT, ROOT);
            printf("Rank[%d] Send curIndex %d success\n", rank, curIndex);         
            send(pixel, curIndex * sizeof(Pixel), MPI_CHAR, ROOT);
            printf("Rank[%d] Send pixel success\n", rank);
        }
        //delete [] pixel;
    }
    if (isEnable && rank == ROOT && size != 1) {
        std::vector<Pixel> v; 
        for (int threads = 1; threads < size; ++threads) {
            int pixelNum;
            recv(&pixelNum, sizeof(int), MPI_INT, threads);
            
            printf("Thread[%d] recv pixelNum =  %d\n", threads, pixelNum);   
            
            Pixel *pixel = new Pixel[pixelNum];
            recv(pixel, pixelNum * sizeof(Pixel), MPI_CHAR, threads);
            
            for (int i = 0; i < pixelNum; ++i)
                v.push_back(pixel[i]);
            delete [] pixel;
        }
       
        std::vector<Pixel>::iterator it;
        int drawTimes = 2;
        while (drawTimes--)
          for (it = v.begin(); it != v.end(); ++it) {
              int i = it->i;
              int j = it->j;
              int repeats = it->repeats;

              XSetForeground (display, gc,  
                          1024 * 1024 * (repeats % 256));	
              XDrawPoint (display, window, gc, i, j);
          }
    }
    end = MPI_Wtime();
    printf("TIMER: rank %d time %.6lf points %d\n", rank, end - start, points);

    MPI_Finalize();
    if (isEnable) sleep(5);
	return 0;
}
