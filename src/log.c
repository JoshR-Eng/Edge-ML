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

// =====================================================================
//                      --- Configuration ---
// =====================================================================

// Need to update these to match Orion Path

#define GPU_LOAD          "..."
#define POWER             "..."
#define LOG_FILE          "..."
#define SAMPLING_RATE_MS  100



// =====================================================================
//                    --- Func. to get values ---
// =====================================================================

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
  FILE *fp = fopen(path, "r");
  if (!fp) return;
  int val = 0;
  if (fscanf(fp, "%d, &val") != 1) val = -1;
  fclose(fp);
  return val;
}


// =====================================================================
//                      --- Main Functions ---
// =====================================================================

int main() {
  printf("Logging Hardware Metrics to: %s\n", LOG_FILE);
  
  // Open log file to write mode
  FILE *log = fopen(LOG_FILE, "w");
  if (!log) { perror("Failed to open log file"); return 1; };

  // CSV Header
  fprint( log, "Timestamp,CPU_Usage,GPU_Usage,Power\n");
  fflush(log);

  // Define Struct variables
  CpuStats cpu_prev, cpu_curr;
  get_cpu_stats(&cpu_prev);

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double start_time = ts.tv_sec + ts.tv_nsec * 1e-9;

  while (1) {
    nanosleep(INTERVAL_MS * 1e6);

    // CPU
    get_cpu_stats(&cpu_curr);
    double cpu_percent = calculate_cpu_usage(&cpu_prev, &cpu_curr);
    cpu_prev = cpu_curr;

    // GPU
       // val is 0-1000 so div. by 10 for %
    int gpu_raw = read_sys_int(GPU_LOAD);
    double gpu_percentage = (gpu_raw >= 0) ? (gpu_raw / 10.0) : 0.0;
    
    // Power
      // measures in milliwatts
    int power_mw = read_sys_int(POWER);

    // Timestamp
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = ts.tv_sec + ts.tv_nsec * 1e-9;


    // Log to file
    fprint(log, "%.4f,%.2f,%.2f,%d\n", now - start_time, cpu_percent,
           gpu_percent, power_mw);
    fflush(log);
  }

  fclose(log);
  return 0;
}

