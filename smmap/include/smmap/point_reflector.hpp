#ifndef POINT_REFLECTOR_HPP
#define POINT_REFLECTOR_HPP

#include <Eigen/Dense>

namespace smmap
{
    class PointReflector
    {
        public:
            PointReflector()
                : PointReflector( 0, 0, 0 )
            {}

            PointReflector( double mid_x, double min_y, double max_y)
                : mid_x_( mid_x )
                , min_y_( min_y )
                , max_y_( max_y )
            {}

            Eigen::Vector3d reflect( const Eigen::Vector3d& in )
            {
                Eigen::Vector3d out = in;
                out[0] = in[0] - 2*( in[0] - mid_x_ );
                return out;
            }

            Eigen::Vector3d reflect( Eigen::Vector3d&& in )
            {
                Eigen::Vector3d out = in;
                out[0] = in[0] - 2*( in[0] - mid_x_ );
                return out;
            }

        private:
            double mid_x_, min_y_, max_y_;
    };
}


#endif // POINT_REFLECTOR_HPP

