/*----------------------------------------------------------------------
NAME:        src/log.c
VERSION:     1.0
DESCRIPTION: A background script to log the Jetson Orions
             power consumption and GPU/CPU utilisation
----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

// --- Configuration ---
// Need to update these to match Orion Path

#define GPU_LOAD          "..."
#define POWER             "..."
#define LOG_FILE          "..."
#define SAMPLING_RATE_MS  100

// --- Metric Func. ---
// CPU Calculation
typedef struct {
  unsigned long long user;
  unsinged long long nice;
  unsigned long long system;
  unsigned long long idle;
} CpuStats;

void get_cpu_stats(CpuStats *s) {
  FILE *fp = fopen("/proc/stat", "r");
  if (!fp) return;
  char label[10];
  fscanf(fp, "%s %llu %llu %llu %llu",
         label, &s->user, &s->nice, &s->system, &s->idle);
  fclose(fp);
}

double calculate_cpu_usage(CpuStats *prev, CpuStats *curr) {
  unsigned long long prev_total = prev->user + prev->nice + prev->system  + prev->idle;
  unsigned long long curr_total = curr->user + curr->nice + curr->system + curr->idle;
  unsigned long long total_diff = curr_total - prev_total;
  unsigned long long idle_diff  = curr->idle - prev->idle;

  if (total_diff == 0) return 0.0;
  return 100.0 * (double)(total_diff - idle_diff) / total_diff;
}


// Get integer values Power, GPU
int read_sys_int(const char *path) { 
  }
