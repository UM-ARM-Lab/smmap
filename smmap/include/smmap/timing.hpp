#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>

namespace smmap
{
    enum StopwatchControl {RESET, READ};

    inline double stopwatch(const StopwatchControl control = READ)
    {
        static std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        if (control == RESET)
        {
            start_time = std::chrono::high_resolution_clock::now();
        }
        const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double>(end_time - start_time).count();
    }
}

#endif // TIMING_HPP
