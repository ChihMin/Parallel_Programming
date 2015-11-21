#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <ctime>

#define ATOI(params, i) params = atoi(argv[i])
#define LOCK(mutex) pthread_mutex_lock(&mutex)
#define UNLOCK(mutex) pthread_mutex_unlock(&mutex)

using namespace std;

int n, C, T, N;
int curPeople;

pthread_mutex_t carMutex;
pthread_mutex_t curPeopleMutex;
pthread_mutex_t passMutex[20];
pthread_mutex_t printMutex;

pthread_cond_t passCond[20];

pthread_cond_t cond;
bool isCarRuning;
bool isCarStop = false;

vector <long> passList;

void *passenger(void *ID) {
  bool isWalking = true;
  long passID = (long)(ID);
  
  while (!isCarStop) {
     
    // Walking arround
    int sleepTime = rand() % 100 + 1;
    isWalking = true;
    usleep(sleepTime);
    
    LOCK(curPeopleMutex);
      LOCK(printMutex); 
        printf("%d'th passenger returns for another ride\n", passID);
      UNLOCK(printMutex);
      
      if (!isCarStop && curPeople < C) {
        // If there's seat to jump into car
        curPeople++;
        //printf("Passenger [%d] : Go on the car, stop walking\n", passID);
        passList.push_back(passID);
        isWalking = false;      
      }
    UNLOCK(curPeopleMutex);
    
    if (!isWalking) { 
      pthread_mutex_lock(&passMutex[passID]);
        //printf("Passenger [%d] : Down from the car, Entering mutex\n", passID);
        pthread_cond_wait(&passCond[passID], &passMutex[passID]);
        isWalking = true;
        //printf("Passenger [%d] : Down from the car, stop walking\n", passID);
      pthread_mutex_unlock(&passMutex[passID]);
      //printf("Passenger [%d] : Mutex unlock from the car, stop walking\n", passID);
    } 
    else {
      LOCK(printMutex); 
        printf("%d'th Passenger Walking arround\n", passID);
      UNLOCK(printMutex);
    }
  }
   
  // printf("Passenger ID : %d Exit\n", passID);
  pthread_exit(NULL);
}

void *car(void *runTimes) {
  int curTime = 0;
  while (N--) {
    //printf("Times %d\n", N);
    
    bool isFull = false;
    while (!isFull) {
      //Waiting people fill the car
      pthread_mutex_lock(&curPeopleMutex);
      if (curPeople >= C)
        isFull = true;
      pthread_mutex_unlock(&curPeopleMutex);
    }

    LOCK(printMutex); 
      printf("car departures at %d millisec. ", curTime, curPeople);
      for (int i = 0; i < passList.size(); ++i) 
        printf("%dth, ", passList[i]); 
      printf("are in the car\n");
    UNLOCK(printMutex);

    usleep(T); // now car Running
    curTime += T;
    
    for (int i = 0; i < passList.size(); ++i) {
      // Broadcast passenger down from car
      long passID = passList[i];
      LOCK(passMutex[passID]);
        pthread_cond_signal(&passCond[passID]);
      UNLOCK(passMutex[passID]);
    } 

    LOCK(printMutex); 
      printf("car arrives at %d millisec. ", curTime);
      for (int i = 0; i < passList.size(); ++i)
        printf("%dth, ", passList[i]); 
      printf("should go off\n");
    UNLOCK(printMutex);
    
    if (N != 0) { 
      LOCK(curPeopleMutex);
        curPeople = 0; // release all people
        passList.clear();
      UNLOCK(curPeopleMutex);
    }
  }
  isCarStop = true;
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
  pthread_mutex_init(&printMutex, NULL);
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
