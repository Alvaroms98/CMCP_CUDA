/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, int n) {

  const float relativeTolerance = 1e-3;
  float prod;
  for(int i = 0; i < n; i++) {
    for (int j=0; j < n; j++){
      prod=0.0;
      for (int k=0; k < n; k++){
        prod += A[i*n+k] * B[k*n+j];
      }
      printf("CPU: %f\nGPU: %f\n", prod,C[i*n+j]);
      float relativeError = (prod - C[i*n+j])/prod;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        printf("TEST FAILED\n\n");
        //exit(0);
      }
    }
  }
  printf("TEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

