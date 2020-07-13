#ifndef _UTILS__HPP_
#define _UTILS__HPP_

#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

namespace utils
{

extern uint8_t colors_vis[][3];

float IntersectionOverUnion(const pcl::PolygonMesh &polymesh1,
                            const pcl::PolygonMesh &polymesh2, 
                            const float resolution=0.005);

}

#endif