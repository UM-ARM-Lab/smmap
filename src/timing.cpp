#include "smmap/timing.hpp"

double smmap::GlobalStopwatch(const StopwatchControl control)
{
    static smmap::Stopwatch global_stopwatch;
    return global_stopwatch(control);
}
