#include <cstdio>
#include <cstdlib>
#include <X11/Xlib.h>
#include <unistd.h>
#include <cmath>
#include <omp.h>
#include <string>
#include <pthread.h>
#include <vector>
#include <cstring>


#define toInt(params, i) params = atoi(argv[i])
#define toDouble(params, i) \
  sscanf(argv[i], "%lf", &params)

#define toBool(params, i, arg) \
  sscanf(argv[i], arg, &params)

#define LOCK(mutex) pthread_mutex_lock(&mutex)
#define UNLOCK(mutex) pthread_mutex_unlock(&mutex)

#define EnableGrid 1

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

#define MAX(x, y) x > y ? x : y
#define MIN(x, y) x < y ? x : y


#define MAXN 1000010
#define MINR 0.00000000001
#define FPS(fps) usleep(1000000/fps)
#define F 30

using namespace std;

struct Node {
  double x, y, vx, vy;
  Node() {};
  Node(double _x, double _y,
        double _vx, double _vy) {
    x = _x, y = _y, vx = _vx, vy = _vy;
  }
};

int BigCounter = 0;

struct Tree {
  vector <Node*> nodes;
  Tree* child[4];
  Tree* parent;

  double startX, startY;
  double length;
  double x, y;
  double mass;
 
  Tree() { 
    memset(child, 0, sizeof(child));  
    parent = NULL;
  }
  
  Tree(double _startX, double _startY, 
             double _length, Tree *_parent) {
    memset(child, 0, sizeof(child)); 
    startX = _startX;
    startY = _startY;
    length = _length;
    parent = _parent;
    mass = parent->mass;
  }
  
  Tree(double _startX, double _startY, 
             double _length, double _mass) {
    memset(child, 0, sizeof(child)); 
    startX = _startX;
    startY = _startY;
    length = _length;
    parent = NULL;
    mass = _mass;
  }
  
  bool inRange(double x, double y,
               double beginX, double endX,
               double beginY, double endY); 
  int getChildIndex(Node *node);
  void push(Node *node);
  void dispatch(); 
  int getNodeNums() ;
  void calMassCenter();
  double getLength();
  double getDistance(const Node *curNode);
  Tree* getChild(int index);
  
  
  ~Tree() {   
    nodes.clear();
    for (int i = 0; i < 4; ++i) { 
      if (child[i])
        delete child[i];
    }
  }
};

const double PI = 3.14159;

GC gc;
Display *display;
Window window;      //initialization for a window
int screen;         //which screen 

Node node[MAXN];

double buildTime = 0, computeTime = 0, IOTime = 0;
double timeBegin, timeEnd; 
int threads, T, N ;
double mass, times;
char *fileName;
bool isEnable, isComplete;
double theta;
double xMin, yMin;
double length;
int XLength;
Tree *root = NULL;

pthread_mutex_t mutex[MAXN];
pthread_mutex_t printMutex; 
 
inline bool Tree::inRange(double x, double y,
             double beginX, double endX,
             double beginY, double endY) {
  if (x < beginX || x > endX) return false;
  if (y < beginY || y > endY) return false;
  return true;
}

inline void Tree::push(Node *node) {  
  nodes.push_back(node);
}

inline double Tree::getLength() {
  return this->length;
}

inline double Tree::getDistance(const Node *curNode) {
  double dx = curNode->x - this->x;
  double dy = curNode->y - this->y;

  if (dx < 0) dx = -dx;
  if (dy < 0) dy = -dy;
  
  return R(dx, dy); 
}

inline int Tree::getNodeNums() {
  return this->nodes.size();
}

inline int Tree::getChildIndex(Node *node) {
  double x = node->x;
  double y = node->y;

  double beginX = this->startX;
  double beginY = this->startY;
  
  int x_bit = (x >= (this->startX + this->length / 2));
  int y_bit = (y >= (this->startY + this->length / 2));
   
  return (x_bit << 1) | (y_bit << 0); 
}

inline Tree* Tree::getChild(int index) {
  return this->child[index];
}

inline void Tree::calMassCenter() {
  int n = nodes.size();
  double totalMass = (double)n * this->mass;
  
  double x_sum = 0, y_sum = 0;
  for (int i = 0; i < n; ++i) {
    x_sum += nodes[i]->x;
    y_sum += nodes[i]->y;
  }
  
  this->x = x_sum * this->mass / totalMass;
  this->y = y_sum * this->mass / totalMass;
}

inline void Tree::dispatch() {
  int n = nodes.size();
  if (n > 1) {
    for (int i = 0; i < n; ++i) {
      Node *ptr = nodes[i];
      //printf("size = %d, x = %lf, y = %lf, beginX = %.10lf, beginY = %.10lf len = %.20lf [%d]\n", n, ptr->x, ptr->y, this->startX, this->startY, this->length, i);
      int childIndex = getChildIndex(ptr);
      if (!child[childIndex]) {
        double beginX = startX + 
                ((childIndex & 2) > 0) * 
                                (this->length / 2);
        
        double beginY = startY + 
                ((childIndex & 1) > 0) * 
                                (this->length / 2);

        child[childIndex] = new Tree(
          beginX, beginY, this->length / 2, this
        );
      }
      //printf("() %d\n", childIndex);
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
inline void printGrid(Tree *root, int step) {
  if (root == NULL) return ;
  if (EnableGrid) { 
    int xPixel = PIXEL(root->startX, xMin);
    int yPixel = PIXEL(root->startY, yMin);
    int lenPixel = XLength * (root->length / length); 
    XSetForeground(display, gc, WhitePixel(display, screen));  // Tell the GC we draw using the RED color
    XDrawRectangle(display, window, gc, xPixel, yPixel, lenPixel, lenPixel);      // Draw the rectangle
  }
  for (int i = 0; i < 4; ++i)
    printGrid(root->getChild(i), step+1); 
}

inline void print() {
    XSetForeground(display,gc,BlackPixel(display,screen));
    XFillRectangle(display,window,gc,0,0,XLength,XLength);

    //printGrid(root, 0);
    for (int i = 0; i < N; ++i) {
      int xPix = PIXEL(node[i].x, xMin);
      int yPix = PIXEL(node[i].y, yMin); 
      
      if (xPix < 0 || xPix >= XLength) continue;
      if (yPix < 0 || yPix >= XLength) continue;
      Draw(xPix, yPix, White);
    } 
    XFlush(display);
    //FPS(5);
}

inline void build_tree(Tree *root, int step) {
  if (root == NULL) 
    return ;
 
  // printf("step = %d\n" ,step);
  //XDrawRectangle(root->startX, root->startY, root->length, root->length, White); 
  
   
  root->dispatch();
  for (int i = 0; i < 4; ++i)
    build_tree(root->getChild(i), step+1); 
}


inline pair<double, double> getAccel(Tree *root, const Node *curNode) {
  int nums = root->getNodeNums();  
  double ax_sum = 0;
  double ay_sum = 0;
  double dx = std::abs(root->x - curNode->x);
  double dy = std::abs(root->y - curNode->y);
  double d = root->length;
  double r = R(dx, dy);
  
  if (r <= MINR) {
    return pair<double, double>(0, 0);
  }
  else if (d/r > theta) {
    //printf("theta = %lf, d/r = %lf\n", theta, d/r);
    for(int i = 0; i < 4; ++i) {
      if (Tree *child = root->getChild(i)) {
        pair<double, double> ret = getAccel(child, curNode);
        ax_sum += ret.first;
        ay_sum += ret.second;   
      }
    }
  }
  else if (nums == 1 || d/r < theta) { 
    // let whole system be a body
    // double totalMass = (double)nums * root->mass; 
    
    double totalMass = root->mass * (double)nums;
    //printf("%lf %lf %lf\n", root->mass, totalMass, (double)nums); 
    double ax = ax(dx, dy, totalMass);
    double ay = ay(dx, dy, totalMass);
     
    ax = ax * ((root->x - curNode->x > 0) ? 1 : -1);
    ay = ay * ((root->y - curNode->y > 0) ? 1 : -1);

    return pair<double, double>(ax, ay);   
  } 
  
  return pair<double, double>(ax_sum, ay_sum);
}

int counter = 0;
inline void thread_func(double beginX, double beginY, double len) {
  int STEPS = T;
  while (STEPS--) {
    // construct tree
    //printf("STEPS = %d %lf %lf %lf\n" ,STEPS, beginX, beginY, len);
    
    double beginX = 1e9, beginY = 1e9;
    double endX = -1e9, endY = -1e9;

    LOCK(printMutex);

    root = new Tree();
    
    for (int i = 0; i < N; ++i) {
      double x = node[i].x;
      double y = node[i].y;
      beginX = MIN(beginX, x);
      beginY = MIN(beginY, y);

      endX = MAX(endX, x);
      endY = MAX(endY, y);
      
      root->push(&node[i]);
    }
    
    root->startX = beginX;
    root->startY = beginY;
    root->length = MAX(endX - beginX, endY - beginY);
    root->mass = mass;
    root->calMassCenter();
    
    
    timeBegin = omp_get_wtime();
    build_tree(root, 0);
    timeEnd = omp_get_wtime();
    buildTime += timeEnd - timeBegin;

    UNLOCK(printMutex);
    // XFlush(display);
    // sleep(5);

    int chunk = N / threads; 
    int i;

    timeBegin = omp_get_wtime();
    #pragma omp parallel num_threads(threads) 
    {
      #pragma omp for schedule(static) nowait
      for (int i = 0; i < N; ++i){
        //printf("i = %d, thread = %d\n", i, omp_get_thread_num()); 
        Node curNode = node[i];
        
        pair<double, double> accel = 
                    getAccel(root, &curNode);
        
        double ax = accel.first;
        double ay = accel.second;

        double vx_next = V(curNode.vx, ax, times);
        double vy_next = V(curNode.vy, ay, times);

        double x_next = curNode.x + vx_next * times;
        double y_next = curNode.y + vy_next * times;
        //double x_next = NEXT(curNode.x, curNode.vx, ax, times);
        //double y_next = NEXT(curNode.y, curNode.vy, ay, times);

        node[i].x = x_next;
        node[i].y = y_next;
        node[i].vx = vx_next;
        node[i].vy = vy_next;
       // printf("[%d], thread:%d, x:%lf, y:%lf, vx:%lf, vy:%lf", i, omp_get_thread_num(), x_next, y_next, vx_next, vy_next);
      }
    }
    // if (isEnable) print(); 
    // release root
    timeEnd = omp_get_wtime();
    computeTime += timeEnd - timeBegin;

    LOCK(printMutex); 
    delete root;
    root = NULL;
    UNLOCK(printMutex);
  }
}

inline void *print(void *ID) {
  while (!isComplete) {
    XSetForeground(display,gc,BlackPixel(display,screen));
    XFillRectangle(display,window,gc,0,0,XLength,XLength);
    
    if (EnableGrid) {
      LOCK(printMutex);
        printGrid(root, 0);
      UNLOCK(printMutex); 
    }
    
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

  double beginX = 1e9, beginY = 1e9;
  double endX = -1e9, endY = -1e9;

  timeBegin = omp_get_wtime();

  FILE *fin = fopen(fileName, "r");
  fscanf(fin, "%d", &N);
  for (int i = 0; i < N; ++i) {
    double x, y, vx, vy;
    beginX = MIN(beginX, x);
    beginY = MIN(beginY, y);

    endX = MAX(endX, x);
    endY = MAX(endY, y);

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

  timeEnd = omp_get_wtime();
  
    
  pthread_mutex_init(&printMutex, NULL);
  pthread_t printer;
  if (isEnable) {
    isComplete = 0;
    pthread_create(&printer, NULL, print, NULL); 
  }
  
  printf("Begin thread function \n");
  thread_func(beginX, beginY, MAX(endX - beginX, endY - beginY));
  printf("End thread function \n");
  
  if (isEnable) {
    isComplete = 1;
    pthread_join(printer, NULL);
  }
  fprintf(stderr, "IO_Time = %.5lf\n", timeEnd - timeBegin);
  fprintf(stderr, "Build_Time = %.5lf\n", buildTime);
  fprintf(stderr, "Compute_Time = %.5lf\n", computeTime);
   
  return 0;
}
