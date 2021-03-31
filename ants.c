#include "acotsp.h"
//library for offsetoff() function
#include "stddef.h"

int main(int argc, char *argv[])
{
	int i, j, k, max_width, max_height, min;
	double start = 0.0, finish = 0.0;
	MPI_Status status;
	MPI_Datatype MPI_CITY, MPI_TABU, MPI_PATH, MPI_ANT;
	
	// Initialize MPI
	MPI_Init(&argc, &argv);
	//Determines the size (&procs) of the group associated with a communicator (MPI_COMM_WORLD)
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	//Determines the rank (&rank) of the calling process in the communicator (MPI_COMM_WORLD)
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Capture the starting time
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	
	if (!rank) {
		ACO_Load_cities(argv[1], &max_width, &max_height);
		printf("Cities: %d\nProcesses: %d\nAnts: %d\nAlpha: %3.2f\nBeta: %3.2f\nRho: %3.2f\nQ: %d\n\n", NUM_CITIES, procs, NUM_ANTS, ALPHA, BETA, RHO, Q);
	}
	
	// Broadcast the cities to all processes
	
	// Creates a contiguous datatype 
	MPI_Type_contiguous(2, MPI_INT, &MPI_CITY);
	MPI_Type_contiguous(NUM_CITIES, MPI_INT, &MPI_TABU);
	MPI_Type_contiguous(NUM_CITIES, MPI_INT, &MPI_PATH);
	
	MPI_Type_commit(&MPI_CITY);
	
	// Create ant struct
	//int city, next_city, tabu[NUM_CITIES], path[NUM_CITIES], path_index;
	//double tour_distance;
    int blocklengths[6] = {1,1, 1, 1, 1, 1};
    MPI_Datatype 	types[6] = {MPI_INT, MPI_INT, MPI_TABU, MPI_PATH, MPI_INT, MPI_DOUBLE};
    MPI_Aint 		offsets[6] = { offsetof( ACO_Ant, city ), offsetof( ACO_Ant, next_city), offsetof( ACO_Ant, tabu), offsetof( ACO_Ant, path ), offsetof( ACO_Ant, path_index ), offsetof( ACO_Ant, tour_distance )};
	
	MPI_Datatype tmp_type;

	MPI_Type_create_struct(6, blocklengths, offsets, types, &tmp_type);
	MPI_Type_create_resized( tmp_type, 0, sizeof(ACO_Ant), &MPI_ANT );
	MPI_Type_commit(&MPI_ANT);
	
	// Broadcasts a message from the process with rank "root" to all other processes of the communicator 
	MPI_Bcast(city, NUM_CITIES, MPI_CITY, 0, MPI_COMM_WORLD);
	MPI_Bcast(pheromone, NUM_CITIES*NUM_CITIES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Construct the city graph and setup/distribute the ants
	ACO_Link_cities();
	ACO_Reset_ants();
	
	MPI_Scatter(ant, 1, MPI_ANT, &antInstance, 1, MPI_ANT, 0, MPI_COMM_WORLD);
	antInstance = ant[rank];

	for(j=0; j<NUM_TOURS*NUM_CITIES; j++) {
		ACO_Step_ants();
		
		//To attempt lower granularity
		/*
		if(j % (NUM_CITIES/2) == 0 && j != 0) {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Gather(&antInstance, 1, MPI_ANT, &ant[rank], 1, MPI_ANT, 0, MPI_COMM_WORLD);
			
			if(!rank){
				ACO_Update_pheromone2();
			}
			
			MPI_Bcast(pheromone, NUM_CITIES*NUM_CITIES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		*/
		
		if(j % NUM_CITIES == 0 && j != 0) {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Gather(&antInstance, 1, MPI_ANT, &ant[rank], 1, MPI_ANT, 0, MPI_COMM_WORLD);
			
			if(!rank){
				ACO_Update_pheromone();
				ACO_Update_best();
				ACO_Reset_ants();
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(pheromone, NUM_CITIES*NUM_CITIES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Scatter(ant, 1, MPI_ANT, &antInstance, 1, MPI_ANT, 0, MPI_COMM_WORLD);
			antInstance = ant[rank];
		}
	}
	
	// Capture the ending time
	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();
	if(!rank) {
		printf("Final Distance (%.15f): %.15f\n", finish-start, best.distance);
		fflush(stdout);
		//ACO_Export_processing(max_width, max_height);
	}
	
	MPI_Type_free(&MPI_PATH);
	MPI_Type_free(&MPI_TABU);
	MPI_Type_free(&MPI_CITY);
	MPI_Type_free(&MPI_ANT);
	MPI_Finalize();
	
	return 0;
}

/** ACO_Step_ants
	Iterates each ant one step in their tours.
	
	@return void
*/
void ACO_Step_ants()
{	
	// path_index = number of cities that the ant has went through 0 - n
	if(antInstance.path_index < NUM_CITIES) {
		antInstance.next_city = ACO_Next_city();  // Pick our next city
		antInstance.tour_distance += distance[antInstance.city][antInstance.next_city];
		antInstance.path[antInstance.path_index++] = antInstance.next_city;
		antInstance.tabu[antInstance.next_city] = 1;
		//printf("Ant %d moved from %d, to %d, tour distance so far: %lf\n", rank, antInstance.city, antInstance.next_city, antInstance.tour_distance);
		antInstance.city = antInstance.next_city; // Move the ant
		
		if(antInstance.path_index == NUM_CITIES) {
			antInstance.tour_distance += distance[antInstance.path[NUM_CITIES-1]][antInstance.path[0]];
		}
		
	}
}

/** ACO_Reset_ants
	Reset each ant's tour information and uniformly distribute each ant accross the city graph.
	
	@return void
*/
void ACO_Reset_ants()
{
	int i, j, uniform = 0;
	
	if(uniform == NUM_CITIES) uniform = 0;
	for(i=0; i<NUM_ANTS; i++) {
		ant[i].city = uniform;
		ant[i].path_index = 1;
		ant[i].tour_distance = 0;
		
		for(j=0; j<NUM_CITIES; j++) {
			ant[i].tabu[j] = 0;
			ant[i].path[j] = -1;
		}
		ant[i].tabu[ant[i].city] = 1;
		ant[i].path[0] = ant[i].city;
		
		uniform++;
	}
}

/** ACO_Next_city
	Chooses which city to visit next based on path pheromones and distances for a specific ant.
	
	@param ant_index [0 .. NUM_ANTS]
	Index of the ant we are deciding the next move for.
	
	@return city index if no devide by zero error.  Else -1.
*/
int ACO_Next_city()
{
	double denominator = 0.0, c = 0.0, r;
	int i;
	struct timeval time;
	
	gettimeofday(&time, 0);
	srandom((int)(time.tv_usec * 1000000 + time.tv_sec)+rank);
	r = (double)random()/(double)RAND_MAX;
	
	//for each city
	for(i=0; i<NUM_CITIES; i++) {
		if(!antInstance.tabu[i]) denominator += ACO_Prob_product(antInstance.city, i);
	}
	
	if(denominator != 0.0) {	
		for(i=0; i<NUM_CITIES; i++) {
			if(!antInstance.tabu[i]) {
				c += ACO_Prob_product(antInstance.city, i)/denominator;
				if(r <= c) break;
			}
		}

		return i;
	} else {
		return -1;
	}
}

/** ACO_Update_best
	Decides which ant has the best tour and stores that information.
	
	@return void
*/
void ACO_Update_best()
{
	int i, j;
	
	for(i=0; i<NUM_ANTS; i++) {
		//printf("(Rank %d) Ant %d got distance: %lf\n", rank, i, ant[i].tour_distance);
		if(ant[i].tour_distance < best.distance || best.distance == 0.0) {
			best.distance = ant[i].tour_distance;
			printf("New best route found: %lf\n\n", best.distance);
			for(j=0; j<NUM_CITIES; j++) best.path[j] = ant[i].path[j];
		}
	}
}

/** ACO_Update_pheromone
	Updates the pheromone trail matrix.  First, it evaporates the current trails,
	and then deposit new pheromone amounts for each ant's tour.
	
	@return void
*/
void ACO_Update_pheromone()
{
	int i, j, from, to;
	
	// Evaporate pheromone
	for(i=0; i<NUM_CITIES; i++) {
		for(j=0; j<NUM_CITIES; j++) {
			if(i != j) {
				pheromone[i][j] *= 1.0-RHO;
				if(pheromone[i][j] < 0.0) pheromone[i][j] = 1.0/NUM_CITIES;
			}
		}
	}

	
	// Deposit pheromone
	for(i=0; i<NUM_ANTS; i++) {
		for(j=0; j<NUM_CITIES; j++) {
			from = ant[i].path[j];
			
			if(j < NUM_CITIES-1) to = ant[i].path[j+1];
			else to = ant[i].path[0];
			
			pheromone[from][to] += Q/ant[i].tour_distance;
			pheromone[to][from] = pheromone[from][to];		
		}
	}
}

//To attempt lower granularity
/*
void ACO_Update_pheromone2()
{
	int i, j, from, to;
	
	// Evaporate pheromone
	for(i=0; i<NUM_CITIES; i++) {
		for(j=0; j<NUM_CITIES; j++) {
			if(i != j) {
				pheromone[i][j] *= 1.0-RHO;
				if(pheromone[i][j] < 0.0) pheromone[i][j] = 1.0/NUM_CITIES;
			}
		}
	}

	
	// Deposit pheromone
	for(i=0; i<NUM_ANTS; i++) {
		for(j=0; j<(NUM_CITIES/2); j++) {
			from = ant[i].path[j];
			
			if(j < (NUM_CITIES/2)-1) to = ant[i].path[j+1];
			else to = ant[i].path[0];
			
			pheromone[from][to] += Q/ant[i].tour_distance;
			pheromone[to][from] = pheromone[from][to];		
		}
	}
}
*/

/** ACO_Load_cities
	Opens a file that contains city coordinates and load that information into 
	the city matrix to be used by the program.
	
	@param filename
	Character array for the filename of the city file.
	
	@param max_width
	Maximum width of the points in the file.
	
	@param max_height
	Maximum height of the points in the file.
	
	@return void
*/
void ACO_Load_cities(char *filename, int *max_width /*out*/, int *max_height /*out*/)
{
	FILE *fp;
	int i;
	
	fp = fopen(filename, "r");
	fscanf(fp, "%dx%d", max_width, max_height);
	for(i=0; i<NUM_CITIES; i++) {
		fscanf(fp, "%d,%d", &city[i].x, &city[i].y);
	}
	
	fclose(fp);
}

/** ACO_Link_cities
	Constructs the fully connected graph of cities by defining the 
	distance and pheromone matrices.  By default pheromone levels are set to
	(1.0 / the number of cities).
	
	@return void
*/
void ACO_Link_cities()
{
	int i, j;
	
	for(i=0; i<NUM_CITIES; i++) {
		for(j=0; j<NUM_CITIES; j++) {
			distance[i][j] = 0.0;
			if(i != j) {
				distance[i][j] = distance[j][i] = ACO_Distance(city[i].x, city[i].y, city[j].x, city[j].y);
			}
			pheromone[i][j] = 1.0/NUM_CITIES;
		}
	}
}

/** ACO_Prob_product
	Calculates the pheromone/distance product for use in the ACO probability function.
	
	@param from [0 .. NUM_CITIES]
	Index of a starting city
	
	@param to [0 .. NUM_CITIES]
	Index of a ending city
	
	@return double value for product
*/
double ACO_Prob_product(int from, int to)
{
	return pow(pheromone[from][to], ALPHA) * pow((1.0/distance[from][to]), BETA);
}

/** ACO_Distance
	Calculates the distance between two points on the cartesian plane.
	
	@param x1
	X component for the first point
	
	@param y1
	Y component for the first point
	
	@param x2
	X component for the second point
	
	@param y2
	Y component for the second point
	
	@return double value for distance
*/
double ACO_Distance(int x1, int y1, int x2, int y2)
{
	return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

/** ACO_Export_processing
	Prints out a script to be pasted into Processing (http://processing.org/) for visualization.
	This can be used if OPENGL and GLUT aren't available.
	
	@param max_width
	Width to use for the size of window.
	
	@param max_height
	Height to use for the size of window.
	
	@return void
*/
void ACO_Export_processing(int max_width, int max_height)
{
	int i, j;
	printf("\nsize(%d, %d);\nbackground(20);\nstroke(60);\nfill(255);\n", max_width, max_height);
	for(i=0; i<NUM_CITIES; i++) {
		if(i<NUM_CITIES-1) printf("line(%d, %d, %d, %d);\n", city[best.path[i]].x, city[best.path[i]].y, city[best.path[i+1]].x, city[best.path[i+1]].y);
		else printf("line(%d, %d, %d, %d);\n", city[best.path[i]].x, city[best.path[i]].y, city[best.path[0]].x, city[best.path[0]].y);
	}
	for(i=0; i<NUM_CITIES; i++) {
		printf("ellipse(%d, %d, 4, 4);\n", city[i].x, city[i].y);
	}
	fflush(stdout);
}
