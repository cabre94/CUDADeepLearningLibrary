// cronometro para procesos en CPU
// USO: cpu_timer T; T.tic();...calculo...;T.tac(); cout << T.ms_elapsed << "ms\n";

#pragma once

#include <ctime>

struct timespec diff(timespec start, timespec end)
{
        timespec temp;
       	if ((end.tv_nsec-start.tv_nsec)<0) {
                temp.tv_sec = end.tv_sec-start.tv_sec-1;
                temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } else {
                temp.tv_sec = end.tv_sec-start.tv_sec;
                temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        }
        return temp;
}

struct cpu_timer{
        struct timespec time1, time2;
	double ms_elapsed;

        cpu_timer(){
        	tic();
        }
       ~cpu_timer(){}

        void tic(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        }
        double tac(){
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
                return(ms_elapsed=elapsed());
        }
        double elapsed(){
            return (double)diff(time1,time2).tv_sec*1000 + (double)diff(time1,time2).tv_nsec*0.000001;
        }
};

#define CRONOMETRAR_CPU( X,VECES ) {  { \
                            cpu_timer t; \
			    float msacum=0.0;\
			    float msacum2=0.0;\
			    for(int n=0;n<VECES;n++){\
			    	t.tic();\
                            	X; t.tac();\
				msacum+=t.ms_elapsed;\
				msacum2+=(t.ms_elapsed*t.ms_elapsed);\
			    }\
			    std::cout << "CPU: " << (msacum/VECES) << " +- " << \
			    (sqrt(msacum2/VECES - msacum*msacum/VECES/VECES)) \
			    << " ms (" << VECES << " veces)\n"; \
                            }}




