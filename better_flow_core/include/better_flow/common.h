#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <functional>
#include <fstream>
#include <vector>
#include <ctime>
#include <new>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <better_flow/datastructures.h>

typedef long int lint;
typedef long long int sll;
typedef unsigned int uint;
typedef unsigned long int ulong;
typedef unsigned long long int ull;


// Version
#define BF_VERSION "1.0"

// Time conversion
#define FROM_SEC(in) ull(1000000000 * (in))
#define FROM_MS(in) ull(1000000 * (in))

// Camera resolution
#define RES_X 480
#define RES_Y 640


// Whether to print debug messages
#ifndef VERBOSE
  #define VERBOSE false
#endif

// Event buffer parameters
#define MAX_TIME_MS 100
#define MAX_EVENT_PER_PX 100

// Primary data structure to store events
class Event;
typedef LinearEventCloudTemplate<Event> LinearEventCloud;
typedef LinearEventPtrsTemplate<Event> LinearEventPtrs;
typedef EventCloudTemplate<CircularArray<Event, MAX_EVENT_PER_PX, FROM_MS(MAX_TIME_MS)>, RES_X, RES_Y> EventCloud;

// Z (time) component of the direction vector
// - can be anything, as long as variables do not overflow
#define NZ 127

// The event timestamp will be divided by T_DIVIDER in integer,
// converted to double and then additionally divided by 10000 
#define T_DIVIDER 1


// Pretty print
#define _header(str) std::string("\033[95m" + std::to_string(str) + "\033[0m")
#define _plain(str) std::string("\033[37m" + std::to_string(str) + "\033[0m")
#define _blue(str) std::string("\033[94m" + std::to_string(str) + "\033[0m")
#define _green(str) std::string("\033[92m" + std::to_string(str) + "\033[0m")
#define _yellow(str) std::string("\033[93m" + std::to_string(str) + "\033[0m")
#define _red(str) std::string("\033[91m" + std::to_string(str) + "\033[0m")
#define _bold(str) std::string("\033[1m" + std::to_string(str) + "\033[0m")
#define _underline(str) std::string("\033[4m" + std::to_string(str) + "\033[0m")


#endif // COMMON_H
