#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <ctime>

#define ATOI(params, i) params = atoi(argv[i])

using namespace std;

int n, C, T, N;
int curPeople;

pthread_mutex_t carMutex;
pthread_mutex_t curPeopleMutex;
pthread_mutex_t passMutex[20];

pthread_cond_t passCond[20];

pthread_cond_t cond;
bool isCarRuning;
bool isCarStop = false;

vector <long> passList;

void *passenger(void *ID) {
  bool isWalking = true;
  long passID = (long)(ID);
  printf("Passenger ID : %d start\n", passID);
  
  while (!isCarStop) {
     
    // Walking arround
    printf("Passenger [%d] : Walking arround\n", passID);
    int sleepTime = rand() % 100 + 1;
    isWalking = true;
    usleep(sleepTime);
    pthread_mutex_lock(&curPeopleMutex);
      if (!isCarStop && curPeople < C) {
        // If there's seat to jump into car
        curPeople++;
        printf("Passenger [%d] : Go on the car, stop walking\n", passID);
        passList.push_back(passID);
        isWalking = false;      
      }
    pthread_mutex_unlock(&curPeopleMutex);
    
    if (!isWalking) { 
      pthread_mutex_lock(&passMutex[passID]);
        printf("Passenger [%d] : Down from the car, Entering mutex\n", passID);
        pthread_cond_wait(&passCond[passID], &passMutex[passID]);
        isWalking = true;
        printf("Passenger [%d] : Down from the car, stop walking\n", passID);
      pthread_mutex_unlock(&passMutex[passID]);
      printf("Passenger [%d] : Mutex unlock from the car, stop walking\n", passID);
    }
  }
   
  printf("Passenger ID : %d Exit\n", passID);
  pthread_exit(NULL);
}

void *car(void *runTimes) {
  int curTime = 0;
  while (N--) {
    printf("Times %d\n", N);
    
    bool isFull = false;
    while (!isFull) {
      //Waiting people fill the car
      pthread_mutex_lock(&curPeopleMutex);
      if (curPeople >= C)
        isFull = true;
      pthread_mutex_unlock(&curPeopleMutex);
    }
     
    printf("car departures at %d millisec. PeupleNum = %d ", curTime, curPeople);
    for (int i = 0; i < passList.size(); ++i) {
      printf("%dth, ", passList[i]);
    } printf("are in the car\n");
    
    usleep(T); // now car Running
    curTime += T;
    printf("car arrives at %d millisec. ", curTime);
    for (int i = 0; i < passList.size(); ++i) {
      long passID = passList[i];
      printf("%dth, ", passID);
      // Broadcast passenger down from car
      pthread_mutex_lock(&passMutex[passID]);
        pthread_cond_signal(&passCond[passID]);
      pthread_mutex_unlock(&passMutex[passID]);
      usleep(10);
    } printf("should go off\n");

    if (N != 0) { 
      pthread_mutex_lock(&curPeopleMutex);
      curPeople = 0; // release all people
      pthread_mutex_unlock(&curPeopleMutex);
      passList.clear();
    }
  }
  isCarStop = true;
  printf("Passenger ID : CAR Exit\n");
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  ATOI(n, 1);
  ATOI(C, 2);
  ATOI(T, 3);
  ATOI(N, 4); 
  curPeople = 0;
  const int NUM_THREADS = n;
  pthread_t passengers[NUM_THREADS];
  pthread_t cars;

  // srand(time(0));
  pthread_mutex_init(&carMutex, NULL);
  pthread_mutex_init(&curPeopleMutex, NULL);
  for (int i = 0 ; i < n; ++i) {
    pthread_cond_init(&passCond[i], NULL);
    pthread_mutex_init(&passMutex[i], NULL);
  }

  for (int i = 0; i < n; i++) {
    pthread_create(&passengers[i], NULL, passenger, (void *)i);
  }
  
  pthread_create(&cars, NULL, car, (void*)N); 
  // join all threads 
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(passengers[i], NULL);
  } pthread_join(cars, NULL);

  pthread_exit(NULL); 
  return 0;
}
