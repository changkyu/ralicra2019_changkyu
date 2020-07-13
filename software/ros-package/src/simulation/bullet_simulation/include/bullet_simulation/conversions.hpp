#ifndef BULLET_CONVERSIONS__H__
#define BULLET_CONVERSIONS__H__

#include <pcl/PolygonMesh.h>
#include <BulletCollision/CollisionShapes/btConvexHullShape.h>

void toBullet(pcl::PolygonMesh::Ptr pclConvHull, btConvexHullShape* btConvHull);

void toBullet( const Eigen::Matrix4f &pose, btTransform &btPose );

#endif