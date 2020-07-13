#include <iostream>

#include <boost/program_options.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/PolygonMesh.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>

#include "bullet_simulation/bullet_simulation.hpp"
#include "bullet_simulation/conversions.hpp"

#include <btBulletDynamicsCommon.h>

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;

Eigen::Matrix4f tf_cam2world;

int main(int argc, char* argv[])
{
    BulletSimulationGui sim;
    //sim.ResetCamera( 0.50,-90,60, 0,0,0 );
    sim.ResetCamera( 0.25,-90,85, 0.12,0,0 );

#if 1
    {
        shape_msgs::Plane table;
        table.coef[0] = 0;
        table.coef[1] = 0;
        table.coef[2] = 1;
        table.coef[3] = -0.28;
        sim.AddPlaneShape( table );
        sim.AddColor(btVector4(0.1,0.1,0.1,1));

        std::vector<float> bucket;
        bucket.push_back(0.4);
        bucket.push_back(0.66);
        bucket.push_back(-0.15);
        bucket.push_back(0.18);
        bucket.push_back(-0.26);
        bucket.push_back(-0.16);
        sim.AddBucketShape(bucket);
        sim.Spin(0.1);
    }
#endif

    shape_msgs::Plane table;
    table.coef[0] = 0;
    table.coef[1] = 0;
    table.coef[2] = 1;
    table.coef[3] = 0;
    sim.AddPlaneShape( table );
    sim.AddColor(btVector4(0.1,0.1,0.1,1));
    
/*
    shape_msgs::Plane draw_right;
    draw_right.coef[0] = 0;
    draw_right.coef[1] = 1;
    draw_right.coef[2] = 0;
    draw_right.coef[3] = -0.1;
    sim.AddPlaneShape( draw_right );

    shape_msgs::Plane draw_down;
    draw_down.coef[0] = 1;
    draw_down.coef[1] = 0;
    draw_down.coef[2] = 0;
    draw_down.coef[3] = 0;
    sim.AddPlaneShape( draw_down );
*/
    btBoxShape* draw_right
     = new btBoxShape(btVector3(1,1,0.075));
    sim.CreateRigidBody(draw_right, btVector3(1,-1-0.11,0.075),0);
    sim.AddColor(btVector4(0.3,0.3,0.3,1));
    btBoxShape* draw_down
     = new btBoxShape(btVector3(1,1,0.075));
    sim.CreateRigidBody(draw_down, btVector3(-1,0,0.075),0);
    sim.AddColor(btVector4(0.3,0.3,0.3,1));

    btRigidBody* body_stick = NULL;

    bool push_stick = true;
    if( strcmp(argv[1],"0")==0 )
    {
        sim.ResetCamera( 0.25,135,15, 0.12,0,0 );

        btBoxShape* box1
         //= new btBoxShape(btVector3(btScalar(0.05),btScalar(0.055),btScalar(0.01)));
         = new btBoxShape(btVector3(btScalar(0.05),btScalar(0.055),btScalar(0.075)));
        //sim.CreateRigidBody(box1, btVector3(0.05,-0.055,0.15),5);
        sim.CreateRigidBody(box1, btVector3(0.05,-0.055,0.075),5);
        sim.AddColor(btVector4(230/255.,  25/255.,  75/255., 1));

        btBoxShape* book
             = new btBoxShape(btVector3(btScalar(0.04),btScalar(0.08),btScalar(0.01)));
            btRigidBody* body_book = sim.CreateRigidBody(book, btVector3(0.20,+0.08,0.01),1);
        sim.AddColor(btVector4(255/255., 225/255.,  25/255., 1));
    }
    else
    {
        sim.ResetCamera( 0.25, 135,15, 0.12,0,0 );

        btCylinderShape* stick
         = new btCylinderShapeZ(btVector3(0.005,0.005,0.25));
        body_stick
         = sim.CreateRigidBody(stick, btVector3(0.22,0.175,0.255),1000);    
        body_stick->setGravity(btVector3(0,0,0));
        body_stick->setMassProps(1000,btVector3(0,0,0));
        sim.AddColor(btVector4(1,1,1,1));

        btBoxShape* box1
         = new btBoxShape(btVector3(btScalar(0.05),btScalar(0.055),btScalar(0.075)));
        sim.CreateRigidBody(box1, btVector3(0.05,-0.055,0.076),5);
        sim.AddColor(btVector4(230/255., 25/255., 75/255.,1));

        if( strcmp(argv[1],"1")==0 )
        {
            btBoxShape* book
             = new btBoxShape(btVector3(btScalar(0.12),btScalar(0.08),btScalar(0.01)));
            btRigidBody* body_book = sim.CreateRigidBody(book, btVector3(0.12,+0.08,0.01),1);
            sim.AddColor(btVector4(255/255., 225/255.,  25/255., 1));
        }
        else if( strcmp(argv[1],"2")==0 )
        {
            btBoxShape* book
             = new btBoxShape(btVector3(btScalar(0.04),btScalar(0.08),btScalar(0.01)));
            btRigidBody* body_book = sim.CreateRigidBody(book, btVector3(0.20,+0.08,0.01),1);
            sim.AddColor(btVector4(255/255., 225/255.,  25/255., 1));
        }

    }    

#if 1

    int idx = 0;
    bool flag = true;  
    double duration = 0.2;  
    double time_cum = 0;
    sim.SpinOnce(0.01);
    while(flag)
    {        
        char filename[256];
        sprintf(filename, "%06d.sim%s.png",idx,argv[1]);
        sim.saveNextFrame(filename);

        int dummy;
        std::cin >> dummy;
        
        sim.SpinOnce(duration);

        if( body_stick )
        {
            if( time_cum > 1.4 )
                body_stick->setLinearVelocity(btVector3(0,0,0));
            else
                body_stick->setLinearVelocity(btVector3(0,-0.5,0));
        }

        time_cum += duration;
        cout << time_cum << endl;

        flag = dummy;
        idx++;
    }
#else
    sim.Spin(0.1);
#endif

}
