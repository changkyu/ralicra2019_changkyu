#ifndef _BULLET_SIMULATION__H__
#define _BULLET_SIMULATION__H__

#include <Eigen/Dense>

#include <pcl/PolygonMesh.h>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletCollision/CollisionShapes/btConvexHullShape.h>

#include "../CommonInterfaces/CommonGUIHelperInterface.h"
#include "../OpenGLWindow/SimpleOpenGL3App.h"
#include "../ExampleBrowser/OpenGLGuiHelper.h"

#include <shape_msgs/Plane.h>

typedef class BulletSimulation
{
public:
    BulletSimulation();    
    virtual ~BulletSimulation();

    virtual void StepSimulation(float deltaTime)
    {
        if (m_dynamicsWorld) m_dynamicsWorld->stepSimulation(deltaTime);        
    }

    //#define DEFAULT_MASS (1.0)
    //#define DEFAULT_FRICTION (0.5)
    #define DEFAULT_MASS (0.01)
    #define DEFAULT_FRICTION (100)

    void AddPlaneShape(      shape_msgs::Plane plane,                             
                             const float friction=DEFAULT_FRICTION         );

    void AddConvexHullShape( pcl::PolygonMesh::Ptr polymesh,                             
                             const btVector3 &position,
                             const float mass=DEFAULT_MASS,
                             const float friction=DEFAULT_FRICTION         );
    void AddConvexHullShape( pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                             const btVector3 &position,
                             const float mass=DEFAULT_MASS,
                             const float friction=DEFAULT_FRICTION         );
    void AddConvexHullShape( btConvexHullShape* shape,
                             const btVector3 &position,
                             const float mass=DEFAULT_MASS,
                             const float friction=DEFAULT_FRICTION         );
    void AddBucketShape(     std::vector<float> bucket,
                             const float mass=DEFAULT_MASS,
                             const float friction=DEFAULT_FRICTION         );

    void InitWorld();
    void ExitWorld();

    btRigidBody* CreateRigidBody( btCollisionShape* shape,
                                  const btVector3 &position,
                                  const float mass=DEFAULT_MASS,
                                  const float friction=DEFAULT_FRICTION    );
    void RemoveLastRigidBody();

    double SpinUntilStable();
    void SetGravity(const btVector3 &gravity);
    
protected:
    btDefaultCollisionConfiguration* m_collisionConfiguration;
    btCollisionDispatcher*  m_dispatcher;
    btBroadphaseInterface*  m_broadphase;
    btConstraintSolver* m_solver;
    btDiscreteDynamicsWorld* m_dynamicsWorld;
    btAlignedObjectArray<btCollisionShape*> m_collisionShapes;
} BulletSimulation;

struct MyOpenGLGuiHelper : public OpenGLGuiHelper
{
    MyOpenGLGuiHelper(struct CommonGraphicsApp* glApp, bool useOpenGL2)
     : OpenGLGuiHelper(glApp, useOpenGL2){}

    virtual ~MyOpenGLGuiHelper(){};

    virtual void autogenerateGraphicsObjects(btDiscreteDynamicsWorld* rbWorld);

    static std::vector<btVector4> colors;
};

typedef class BulletSimulationGui : public BulletSimulation
{
public:
    BulletSimulationGui();    
    virtual ~BulletSimulationGui();

    virtual void StepSimulation(float deltaTime);

    void Spin(float speed);
    void SpinOnce(float duration=0.1);

    void ResetCamera( float dist=1, float pitch=180, float yaw=30, 
                      float x=0, float y=0, float z=0              );

    void AddColor(btVector4 color);

    void saveNextFrame(const char* filename)
    {
        m_app->dumpNextFrameToPng(filename);
    }

    void SpinInit();
    void SpinExit();
private:        

    SimpleOpenGL3App* m_app;
    //struct GUIHelperInterface* m_gui;
    struct MyOpenGLGuiHelper* m_gui;

    bool b_init_gui;

} BulletSimulationGui;

#endif