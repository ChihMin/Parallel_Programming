#include <cstdio>
#include <cstdlib>
#include <X11/Xlib.h>
#include <unistd.h>
#include <cmath>
#include <omp.h>
#include <string>
#include <pthread.h>

#define toInt(params, i) params = atoi(argv[i])
#define toDouble(params, i) \
  sscanf(argv[i], "%lf", &params)

#define toBool(params, i, arg) \
  sscanf(argv[i], arg, &params)

#define LOCK(mutex) pthread_mutex_lock(&mutex)
#define UNLOCK(mutex) pthread_mutex_unlock(&mutex)

#define G (6.67384*pow(10, -11))
#define powOfR(dx, dy) (pow(dx, 2)+pow(dy, 2))
#define R(dx, dy) sqrt(powOfR(dx, dy)) 
#define a(dx, dy, m) (G*m / powOfR(dx, dy)) 
#define SIN(dx, dy) (dy / R(dx, dy))
#define COS(dx, dy) (dx / R(dx, dy))
#define ay(dx, dy, m) \
  (a(dx, dy, m) * SIN(dx, dy))
#define ax(dx, dy, m) \
  (a(dx, dy, m) * COS(dx, dy))
#define V(V0, a, t) (V0 + a*t)
#define NEXT(x, V0, a, t) x + V(V0, a, t) * t
#define PIXEL(v, vMin) \
  ((double)XLength * (v - vMin) / length) 

#define Draw(x, y, color) \
  XSetForeground(display, gc, color##Pixel(display,screen)); \
  XDrawPoint (display, window, gc, x, y);

#define Erase(x, y) \
  Draw(x, y, Black);

#define MAXN 1000010
#define MINR 0.01
#define FPS(fps) usleep(1000000/fps)
#define F 20

using namespace std;

struct Node {
  double x, y, vx, vy;
  Node() {};
  Node(double _x, double _y,
        double _vx, double _vy) {
    x = _x, y = _y, vx = _vx, vy = _vy;
  }
};

const double PI = 3.14159;

GC gc;
Display *display;
Window window;      //initialization for a window
int screen;         //which screen 

Node node[MAXN];

int threads, T, N ;
double mass, times;
char *fileName;
bool isEnable, theta, isComplete;
double xMin, yMin;
double length;
int XLength;

pthread_mutex_t mutex[MAXN];

void print();

inline void initGraph(int width,int height)
{
	/* open connection with the server */
    width = height = XLength; 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}

	screen = DefaultScreen(display);

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;
                  
	/* create window */
	window = XCreateSimpleWindow(
        display, 
        RootWindow(display, screen), 
        x, y, width, height, border_width, 
        BlackPixel(display, screen), 
        WhitePixel(display, screen)
    );
	
	/* create graph */
	XGCValues values;
	long valuemask = 0;
	
	gc = XCreateGC(display, window, valuemask, &values);
	//XSetBackground (display, gc, WhitePixel (display, screen));
	XSetForeground (display, gc, BlackPixel (display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
                
	/* draw rectangle */
	XSetForeground(display,gc,BlackPixel(display,screen));
	XFillRectangle(display,window,gc,0,0,width,height);
	XFlush(display);
}

void thread_func() {
  int STEPS = T;
  long numPerThreads = N / threads;
  long i, j; 

while (STEPS--) {
  //printf("STEPS %d\n", STEPS);
#pragma omp parallel num_threads(threads) \
                      shared(node) private(i, j) 
  {
    #pragma omp for schedule(static) nowait 
    for (i = 0; i < N; ++i) {
      double ax_sum = 0;
      double ay_sum = 0;
      long threadID = omp_get_thread_num();
      //fprintf(stderr, "ID = %d\n", threadID);
      
      // printf("[OUT] num of threads = %d\n", omp_get_num_threads());

      Node n1 = node[i];
      //printf("BEFORE ---> ID[%d, %d] = (%lf, %lf, %lf, %lf)\n", threadID, i, node[i].x, node[i].y, node[i].vx, node[i].vy);
      
      // omp_set_nested(true);
      //#pragma omp parallel num_threads(threads)
      { 
      
      // printf("[IN] num of threads = %d\n", omp_get_num_threads());
      //#pragma omp for schedule(dynamic, numPerThreads) nowait 
      for (j = 0; j < N; ++j) {
        if (i == j) continue;
        double x1, y1;
        double x2, y2;
        double ax ,ay;
        double dx, dy;

        Node n2 = node[j];

        x1 = n1.x;
        y1 = n1.y;
        
        x2 = n2.x;
        y2 = n2.y;
         
        dx = x2 > x1 ? x2 - x1 : x1 - x2;
        dy = y2 > y1 ? y2 - y1 : y1 - y2;
        
        double dis = R(dx, dy);
        if (dis < MINR) { 
          //printf("too small\n");
          continue;
        }

        ax = ax(dx, dy, mass);
        ay = ay(dx, dy, mass);

        if (x2 < x1) ax = -ax;
        if (y2 < y1) ay = -ay;
        
        //printf("(%d <-> %d)(%.30lf,%.30lf)\n", i, j, ax, ay);
         
        ax_sum += ax;
        ay_sum += ay; 
      }

      }
      double x_next = NEXT(n1.x, n1.vx, ax_sum, times);
      double y_next = NEXT(n1.y, n1.vy, ay_sum, times);
      double Vx = V(n1.vx, ax_sum, times);
      double Vy = V(n1.vy, ay_sum, times);
      //printf("AFTER  ---> ID[%d, %d] = (%lf, %lf, %lf, %lf, %.30lf, %.30lf)\n\n", threadID, i, x_next, y_next, Vx, Vy, ax_sum, ay_sum);
      
      node[i] = Node(x_next, y_next, Vx, Vy);
    
    }

  }
  }

}

void *print(void *ID) {
  while (!isComplete) {
    XSetForeground(display,gc,BlackPixel(display,screen));
    XFillRectangle(display,window,gc,0,0,XLength,XLength);
    for (int i = 0; i < N; ++i) {
      Node print = node[i];
      int xPix = PIXEL(print.x, xMin);
      int yPix = PIXEL(print.y, yMin); 
      
      if (xPix < 0 || xPix >= XLength) continue;
      if (yPix < 0 || yPix >= XLength) continue;
      Draw(xPix, yPix, White);
    } 
    XFlush(display);
    FPS(F);
  }
  pthread_exit(NULL);
}

int main(int argc,char *argv[]){
  toInt(threads, 1);
  toDouble(mass, 2); 	
  toInt(T, 3);
  toDouble(times, 4);
  fileName = argv[5];
  toDouble(theta, 6);
  
  printf("BEGIN : (%lf %d %lf)\n", mass, T, times); 
   
  string Enable = argv[7];
  isEnable = (Enable == "enable");
   
  if (isEnable) {
    toDouble(xMin, 8);
    toDouble(yMin, 9);
    toDouble(length, 10);
    toInt(XLength, 11);
    initGraph(500, 500);
    printf("BEGIN : (xmin = %lf, ymin = %lf)\n", xMin, yMin); 
  }

    
  FILE *fin = fopen(fileName, "r");
  fscanf(fin, "%d", &N);
  for (int i = 0; i < N; ++i) {
    double x, y, vx, vy;
    fscanf(fin, "%lf %lf %lf %lf", &x, &y, &vx, &vy);
    node[i] = Node(x, y, vx, vy);
    long long int  xPix = (long long int )PIXEL(x, xMin);
    long long int yPix = (long long int)PIXEL(y, yMin); 
    // printf("START -> (%lld %lld) (%lf, %lf) %lf %lf\n", xPix, yPix, node[i].x, node[i].y,  node[i].vx, node[i].vy);   
    if (isEnable) { 
      Draw(xPix, yPix, White);
      XFlush(display);
    }
  }
  pthread_t printer;
  if (isEnable) {
    isComplete = 0;
    pthread_create(&printer, NULL, print, NULL); 
  }
  
  thread_func();
  
  if (isEnable) {
    isComplete = 1;
    pthread_join(printer, NULL);
  }
   
  return 0;
}
