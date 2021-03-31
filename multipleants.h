#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#ifdef OPENGL
#include <GLUT/glut.h> 		// <GL/glut.h> for Linux
#endif

/* Constants */
#define Q 80				// Influences the amount of pheromone deposited
#define RHO 0.2				// Influences the rate of pheromone evaporation
#define ALPHA 1.0			// Influences the appeal of pheromone on a path
#define BETA 2.0			// Influences the appeal of distance of a path

#define NUM_CITIES 1000		// Number of cities
//Ants should be divisible by the number of processes, eg 24 % 12 = 0
#define NUM_ANTS 24			// Number of ants
//Local ants should be divisible by the number of processes
#define LOCAL_ANTS NUM_ANTS/12 //Number of ants per process
#define NUM_TOURS 35		// Number of tours done before every communication
#define NUM_COMMS 0 			// Number of communications (gather, bcast) to do

/* Structs */
typedef struct {
	int city, next_city, tabu[NUM_CITIES], path[NUM_CITIES], path_index;
	double tour_distance;
} ACO_Ant;

typedef struct {
	int x, y;
} ACO_City;

typedef struct { 
	double distance;
	int path[NUM_CITIES];
} ACO_Best_tour;

/* (Local) Global Variables */
ACO_Ant ant[NUM_ANTS];
ACO_Ant localAnts[LOCAL_ANTS];
ACO_Ant antInstance;
ACO_City city[NUM_CITIES];
ACO_Best_tour best, *all_best;
double distance[NUM_CITIES][NUM_CITIES], pheromone[NUM_CITIES][NUM_CITIES];
int rank, procs;

/* Function Definitions */
void ACO_Step_ants();
void ACO_Reset_ants();
void ACO_Update_pheromone();
void ACO_Update_best();
void ACO_Load_cities(char *filename, int *max_width, int *max_height);
void ACO_Link_cities();
int main(int argc, char * argv[]);
int ACO_Next_city();
double ACO_Prob_product(int from, int to);
double ACO_Distance(int x1, int y1, int x2, int y2);
void ACO_Build_best();
void ACO_Display();
void ACO_Export_processing(int max_width, int max_height);
