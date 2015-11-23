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
#define MINR 0.0001
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

struct Tree {
  vector <Node*> nodes;
  Tree* child[4];
  Tree* parent;

  double startX, startY;
  double length;
  double x, y;
  double mass;
 
  
  Tree() { 
    memset(child, NULL, sizeof(child));  
    parent = NULL;
  }
  
  Tree(int _startX, int _startY, 
             int _length, Tree *_parent) {
    memset(child, 
    startX = _startX;
    startY = _startY;
    length = _length;
    parent = _parent;
  }
   
  bool inRange(double x, double y,
               double beginX, double endX,
               double beginY, double endY); 
  int getChildIndex(Node *node);
  void push(Node *node);
  void dispatch(); 
  int getNum() ;
  void calMassCenter();
  Tree* getChild(int index);
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
  
bool Tree::inRange(double x, double y,
             double beginX, double endX,
             double beginY, double endY) {
  
  return false;
}

void Tree::push(Node *node) {  
  nodes.push_back(node);
}

int Tree::getChildIndex(Node *node) {
  double x = node->x;
  double y = node->y;

  int beginX = this->startX;
  int beginY = this->startY;
  
  int x_bit = (x >= (startX + length / 2));
  int y_bit = (y >= (startY + length / 2));
  return (x_bit << 1) | (y_bit << 0); 
}

Tree* Tree::getChild(int index) {
  return this->child[index];
}

void Tree::calMassCenter() {
  int n = nodes.size();
  double totalMass = (double)n * this->mass;
  
  double x_sum = 0, y_sum = 0;
  for (int i = 0; i < n; ++i) {
    x_sum += nodes[i]->x;
    y_sum += nodex[i]->y;
  }

  this->x = x_sum * this->mass / totalMass;
  this->y = y_sum * this->mass / totalMass;
}

void Tree::dispatch() {
  if (nodes.size() > 1) {
    int n = nodes.size();
    for (int i = 0; i < n; ++i) {
      Node *ptr = node[i];
      int childIndex = getChildIndex(ptr);
      if (!child[childIndex]) {
        double beginX = startX + ((childIndex & 2) > 0)*(length / 2);
        double beginY = startY + ((childIndex & 1) > 0)*(length / 2);
        child[childIndex] = new Tree(beginX, beginY, length / 2, this);
      }
      child[childIndex]->push(ptr);                 
    }
    
    // calculate child mass center
    for (int i = 0; i < 4; ++i) {
      if (child[i]) {
        child[i]->calMassCenter();
      }
    }
  }
}

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
  while (STEPS--) {
    // construct tree
    
    
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
