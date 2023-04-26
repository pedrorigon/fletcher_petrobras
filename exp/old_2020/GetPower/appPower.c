// #include <stdio.h>
// #include <stdlib.h>
// #include <sys/types.h>
// #include <sys/wait.h>
// #include <unistd.h>
// #include "nvml.h"
// #include <omp.h>
// #include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main(int argc, char **argv){
	if(argc < 2){
		// int i; for(i = 0; i < argc; i++) printf("%s\n", argv[i]);
		fprintf(stderr, "Usage: %s <program>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	printf("PID: %d\n", getpid());
	long SAMPLES_PER_SEC = 20, INTERVAL = 1E9 / SAMPLES_PER_SEC;
	const struct timespec tpower = {(long) 0, (long) INTERVAL};
	char str[41];
	sprintf(str, "getPower >> /tmp/power.%d", getpid());

	long int i, value, max = 0;
	double energy = 0, power = 0;
	
	int end, status;
	pid_t childID;

	if((childID = fork()) == -1){
		perror("fork error");
		exit(EXIT_FAILURE);
	}else if(childID == 0){
		printf("PID child: %d\n", getpid());
		execvp(argv[1], &argv[1]);
		printf("execvp returned unexpectedly");
		exit(EXIT_FAILURE);
	}else{
		for(max = 1; max <= 3; max++){
			system(str);
			nanosleep(&tpower, NULL);
		}
		while(1){
			system(str);
			nanosleep(&tpower, NULL);
			max++;
			end = waitpid(childID, &status, WNOHANG|WUNTRACED);
			if(end == childID && WIFEXITED(status)){
			    break;
			}
		}

                sprintf(str, "wc -l /tmp/power.%d | awk {'print $1'} > /tmp/power.lines", getpid());
		system(str);
		FILE *f = fopen("/tmp/power.lines", "r");
                if(f == NULL){
                        perror("file open error");
                        exit(EXIT_FAILURE);
                }
		fscanf(f, "%ld", &max);
		fclose(f);

		sprintf(str, "/tmp/power.%d", getpid());
		f = fopen(str, "r");
		if(f == NULL){
			perror("file open error");
			exit(EXIT_FAILURE);
		}
		float dt = (float)INTERVAL / (float)1E9;
		double b = (double) (max - 1) * dt;
		long first, last;

		printf("max=%ld %ld %f\n", max, INTERVAL, (float)max * dt);
		fscanf(f, "%ld\n", &first);
		for(i = 1; i < max - 3; i++){
			fscanf(f, "%ld\n", &value);
			energy += (double) value;
			power += (double) value;
		}
		fscanf(f, "%ld\n", &last);
		power = (power + first + last) / (float)max;
		energy = (b / (2 * (max-1))) * (first + 2 * energy + last);
		fprintf(stderr, "%.2lf\n", power);
                fprintf(stderr, "%.2lf\n", energy);
		
		printf("power: %lf Watts\n", power);
		printf("energy (average):   %lf Joules\n", power * (float)max * dt);
		printf("energy (integrate): %lf Joules\n", energy);

		fclose(f);
	}
}
