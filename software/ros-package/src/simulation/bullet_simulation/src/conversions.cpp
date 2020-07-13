#include <pcl_conversions/pcl_conversions.h>

#include "bullet_simulation/conversions.hpp"

using namespace std;
using namespace pcl;

void toBullet( PolygonMesh::Ptr pclConvHull, btConvexHullShape* btConvHull )
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromPCLPointCloud2(pclConvHull->cloud, cloud);
    for( size_t p=0; p<cloud.size(); p++ )
    {
        btConvHull->addPoint( btVector3(cloud[p].x,cloud[p].y,cloud[p].z) );
    }    
}

void toBullet( const Eigen::Matrix4f &pose, btTransform &btPose )
{
    Eigen::Quaternionf q(pose.block<3,3>(0,0));
    q.normalize();
    btQuaternion btQ(q.x(), q.y(), q.z(), q.w());
    
    btPose.setIdentity();
    btPose.setOrigin(btVector3(pose(0,3),pose(1,3),pose(2,3)));
    btPose.setRotation(btQ);
}