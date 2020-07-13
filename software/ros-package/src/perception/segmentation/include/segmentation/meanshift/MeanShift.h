#pragma once 

#include <vector>

struct Cluster {
    std::vector<double> mode;
    std::vector<std::vector<double> > original_points;
    std::vector<std::vector<double> > shifted_points;
};

class MeanShift {
public:
    typedef std::vector<double> Point;

    MeanShift() { set_kernel(NULL); }
    MeanShift(double (*_kernel_func)(double,double)) { set_kernel(kernel_func); }
    void meanshift(const std::vector<Point> & points,
                   std::vector<Point> &shifted_points, 
                   double kernel_bandwidth,
                   double EPSILON = 0.00001);
    void cluster(const std::vector<Point> &, double, 
                 std::vector<Cluster> &, 
                 std::vector<int> &);

private:
    double (*kernel_func)(double,double);
    void set_kernel(double (*_kernel_func)(double,double));
    void shift_point(const Point&, const std::vector<Point> &, double, Point&);
    void cluster(const std::vector<Point> &, const std::vector<Point> &,
                 std::vector<Cluster> &clusters, std::vector<int> &labels);
};
