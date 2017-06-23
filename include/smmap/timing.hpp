#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>

namespace smmap
{
    enum StopwatchControl {RESET, READ};

    class Stopwatch
    {
        public:
            Stopwatch()
                : start_time_(std::chrono::high_resolution_clock::now())
            {}

            double operator() (const StopwatchControl control = READ)
            {
                const auto end_time = std::chrono::high_resolution_clock::now();
                if (control == RESET)
                {
                    start_time_ = end_time;
                }

                return std::chrono::duration<double>(end_time - start_time_).count();
            }

        private:
            std::chrono::high_resolution_clock::time_point start_time_;
    };

    double GlobalStopwatch(const StopwatchControl control = READ);
}

#endif // TIMING_HPP
