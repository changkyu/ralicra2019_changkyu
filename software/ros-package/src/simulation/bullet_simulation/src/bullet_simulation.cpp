#include <iostream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <BulletCollision/CollisionShapes/btConvexHullShape.h>

#include "../Utils/b3Clock.h"

#include "bullet_simulation/conversions.hpp"
#include "bullet_simulation/bullet_simulation.hpp"

using namespace std;

static
btConvexHullShape* ReduceConvexShape(btConvexHullShape* originalConvexShape )
{
    //create a hull approximation
    btShapeHull* hull = new btShapeHull(originalConvexShape);
    btScalar margin = originalConvexShape->getMargin();
    hull->buildHull(margin);
    return new btConvexHullShape((const btScalar *)hull->getVertexPointer(),hull->numVertices());
}

BulletSimulation::BulletSimulation()
{
    ///collision configuration contains default setup for memory, collision setup
    m_collisionConfiguration = new btDefaultCollisionConfiguration();
    //m_collisionConfiguration->setConvexConvexMultipointIterations();

    ///use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see Extras/BulletMultiThreaded)
    m_dispatcher = new  btCollisionDispatcher(m_collisionConfiguration);

    m_broadphase = new btDbvtBroadphase();

    ///the default constraint solver. For parallel processing you can use a different solver (see Extras/BulletMultiThreaded)
    m_solver = new btSequentialImpulseConstraintSolver;        

    m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);

    SetGravity(btVector3(0, 0, -10));
}   

BulletSimulation::~BulletSimulation()
{
    delete m_dynamicsWorld;
    m_dynamicsWorld=0;

    delete m_solver;
    m_solver=0;

    delete m_broadphase;
    m_broadphase=0;

    delete m_dispatcher;
    m_dispatcher=0;

    delete m_collisionConfiguration;
    m_collisionConfiguration=0;
}

void BulletSimulation::SetGravity(const btVector3 &gravity)
{
    m_dynamicsWorld->setGravity(gravity);
}

void BulletSimulation::InitWorld()
{
    
#if 0    
    /////////////////////////////
    // Create the ground plane //
    /////////////////////////////
    btBoxShape* groundShape
     = new btBoxShape(btVector3(btScalar(2),btScalar(2),btScalar(2)));
    m_collisionShapes.push_back(groundShape);
    //btTransform groundTransform;
    //groundTransform.setIdentity();
    //groundTransform.setOrigin(btVector3(0,0,-50));
    //groundTransform.setOrigin(btVector3(0,0,-3));
    CreateRigidBody(groundShape,btVector3(0,0,-2-1),0);
#endif 
    
#if 0

    btBoxShape* tableShape
     = new btBoxShape(btVector3(btScalar(0.25),btScalar(0.50),btScalar(0.72/2)));
    // = new btBoxShape(btVector3(btScalar(50),btScalar(50),btScalar(0.50)));
    m_collisionShapes.push_back(tableShape);
    //btTransform tableTransform;
    //tableTransform.setIdentity();
    //tableTransform.setOrigin(btVector3(0.75,0,0.72/2-1));
    //tableTransform.setOrigin(btVector3(0,0,-0.50));
    CreateRigidBody(tableShape,btVector3(0.75,0,0.72/2-1),100);

    btBoxShape* colShape
     = new btBoxShape(btVector3(.1,.1,.1));
    m_collisionShapes.push_back(colShape);
    btTransform startTransform;
    startTransform.setIdentity();
    startTransform.setOrigin(btVector3(0,0,1));
    CreateRigidBody(colShape,startTransform);

    // tmp
    btBoxShape* colShape
     = new btBoxShape(btVector3(.1,.1,.1));
    m_collisionShapes.push_back(colShape);
    btTransform startTransform;
    startTransform.setIdentity();
    btScalar mass(1.f);
    btVector3 localInertia(0,0,0);
    colShape->calculateLocalInertia(mass,localInertia);
    for (int k=0;k<5;k++)
    {
        for (int i=0;i<5;i++)
        {
            for(int j = 0;j<5;j++)
            {
                startTransform.setOrigin(btVector3(
                                        btScalar(0.2*i),
                                        btScalar(2+.2*k),
                                        btScalar(0.2*j)));

        
                CreateRigidBody(mass,startTransform,colShape);
            }
        }
    }
#endif
}

void BulletSimulation::ExitWorld()
{    
    if (m_dynamicsWorld)
    {
        int i;
        for (i = m_dynamicsWorld->getNumConstraints() - 1; i >= 0; i--)
        {
            m_dynamicsWorld->removeConstraint(m_dynamicsWorld->getConstraint(i));
        }
        for (i = m_dynamicsWorld->getNumCollisionObjects() - 1; i >= 0; i--)
        {
            btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
            btRigidBody* body = btRigidBody::upcast(obj);
            if (body && body->getMotionState())
            {
                delete body->getMotionState();
            }
            m_dynamicsWorld->removeCollisionObject(obj);
            delete obj;
        }
    }
    //delete collision shapes
    for (int j = 0; j<m_collisionShapes.size(); j++)
    {
        btCollisionShape* shape = m_collisionShapes[j];
        delete shape;
    }
    m_collisionShapes.clear();
}

btRigidBody* 
BulletSimulation::CreateRigidBody( btCollisionShape* shape,                                    
                                   const btVector3 &position,
                                   const float mass,
                                   const float friction            )
{
    btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

    m_collisionShapes.push_back(shape);
    /*
    if( shape->getNumPoints() > 100 )
    {
        shape = ReduceConvexShape(shape);
        m_collisionShapes.push_back(shape);
    } 
    */

    btVector3 localInertia(0,0,0);
    if( mass!=0.f ) shape->calculateLocalInertia(mass, localInertia);

    //btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
    btTransform transform;
    transform.setIdentity();
    transform.setOrigin(position);
    btDefaultMotionState* noMotion = new btDefaultMotionState(transform);

    btRigidBody::btRigidBodyConstructionInfo 
      cInfo(mass, noMotion, shape, localInertia);

    btRigidBody* body = new btRigidBody(cInfo);    
    body->setUserIndex(-1);
    body->setFriction(friction);
    body->setRestitution(0);
    //body->setDamping(1,1);
    m_dynamicsWorld->addRigidBody(body);
    return body;
}

void BulletSimulation::RemoveLastRigidBody()
{
    if (m_dynamicsWorld)
    {
        int i = m_dynamicsWorld->getNumCollisionObjects() - 1;
        btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
        btRigidBody* body = btRigidBody::upcast(obj);
        if (body && body->getMotionState())
        {
            delete body->getMotionState();
        }
        m_dynamicsWorld->removeCollisionObject(obj);
        delete obj;        
    }
    //delete collision shapes
    if( m_collisionShapes.size() > 0 )
    {
        int i = m_collisionShapes.size()-1;
        btCollisionShape* shape = m_collisionShapes[i];
        delete shape;
        m_collisionShapes.pop_back();
    }    
}

void BulletSimulation::AddPlaneShape( shape_msgs::Plane plane,                                      
                                      const float friction            )
{
    btStaticPlaneShape* shape = new btStaticPlaneShape(
        btVector3(plane.coef[0],plane.coef[1],plane.coef[2]), 
        btScalar(plane.coef[3]) );
    CreateRigidBody(shape,btVector3(0,0,0),0,friction);
}

void BulletSimulation::AddConvexHullShape( pcl::PolygonMesh::Ptr polymesh,
                                           const btVector3 &position,
                                           const float mass,
                                           const float friction            )
{
    btConvexHullShape* shape = new btConvexHullShape();
    shape->setMargin(0);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromPCLPointCloud2(polymesh->cloud, cloud);
    for( size_t p=0; p<cloud.size(); p++ )
    {
        shape->addPoint(btVector3(cloud[p].x,cloud[p].y,cloud[p].z));
    }
        
    AddConvexHullShape(shape, position, mass, friction);
}

void BulletSimulation::AddConvexHullShape( pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                                           const btVector3 &position,
                                           const float mass,
                                           const float friction            )
{
    btConvexHullShape* shape = new btConvexHullShape();
    shape->setMargin(0);
    
    for( size_t p=0; p<cloud.size(); p++ )
    {
        shape->addPoint(btVector3(cloud[p].x,cloud[p].y,cloud[p].z));
    }
        
    AddConvexHullShape(shape, position, mass, friction);
}

void BulletSimulation::AddConvexHullShape( btConvexHullShape* shape,
                                           const btVector3 &position,
                                           const float mass,
                                           const float friction            )
{
    CreateRigidBody(shape,position,mass,friction);
}

void BulletSimulation::AddBucketShape( std::vector<float> bucket,                                       
                                       const float mass,
                                       const float friction        )
{
    float width = 0.01;

    btBoxShape* front   = new btBoxShape(
        btVector3(width, (bucket[3]-bucket[2])/2, (bucket[5]-bucket[4])/2));
    btBoxShape* back    = new btBoxShape(
        btVector3(width, (bucket[3]-bucket[2])/2, (bucket[5]-bucket[4])/2));
    btBoxShape* left    = new btBoxShape(
        btVector3((bucket[1]-bucket[0])/2, width, (bucket[5]-bucket[4])/2));
    btBoxShape* right   = new btBoxShape(
        btVector3((bucket[1]-bucket[0])/2, width, (bucket[5]-bucket[4])/2));
    btBoxShape* bottom  = new btBoxShape(
        btVector3((bucket[1]-bucket[0])/2, (bucket[3]-bucket[2])/2, 0.01));

    btVector3 position( (bucket[1]+bucket[0])/2,
                        (bucket[3]+bucket[2])/2,
                        (bucket[5]+bucket[4])/2  );

    btTransform tf_front;
    tf_front.setIdentity();
    tf_front.setOrigin(  btVector3(bucket[0]-position[0]-width,0,0));
    btTransform tf_back;
    tf_back.setIdentity();
    tf_back.setOrigin(   btVector3(bucket[1]-position[0]+width,0,0));
    btTransform tf_left;
    tf_left.setIdentity();
    tf_left.setOrigin(   btVector3(0,bucket[2]-position[1]-width,0));
    btTransform tf_right;
    tf_right.setIdentity();
    tf_right.setOrigin(  btVector3(0,bucket[3]-position[1]+width,0));
    btTransform tf_bottom;
    tf_bottom.setIdentity();
    tf_bottom.setOrigin( btVector3(0,0,bucket[4]-position[2]-width));

    btCompoundShape* shape = new btCompoundShape();
    shape->addChildShape(tf_front,  front);
    shape->addChildShape(tf_back,   back);
    shape->addChildShape(tf_left,   left);
    shape->addChildShape(tf_right,  right);    
    shape->addChildShape(tf_bottom, bottom);    
    
    CreateRigidBody(shape,position,mass,friction);    
}

static double compute_dist(btQuaternion &q1, btQuaternion &q2)
{
    btQuaternion q11 = q1.normalize();
    btQuaternion q22 = q2.normalize();

    return 1 - (q11.x()*q22.x() + q11.y()*q22.y() + q11.z()*q22.z() + q11.w()*q22.w())*
               (q11.x()*q22.x() + q11.y()*q22.y() + q11.z()*q22.z() + q11.w()*q22.w());
}

static double compute_dist(btVector3 &vec1, btVector3 &vec2)
{
    return sqrt( (vec1[0] - vec2[0]) * (vec1[0] - vec2[0]) +
                 (vec1[1] - vec2[1]) * (vec1[1] - vec2[1]) +
                 (vec1[2] - vec2[2]) * (vec1[2] - vec2[2])   );
}

double BulletSimulation::SpinUntilStable()
{    
    const double eps = 0.0001;
    vector<btVector3> locations_init;
    vector<btVector3> locations_prev;
    vector<btQuaternion> rotations_init;
    vector<btQuaternion> rotations_prev;

    int iter=0;
    double dist_sum=0;
    double dist_local=0;
    do
    {
        dist_local=0;
        for (int o=0; o<m_dynamicsWorld->getNumCollisionObjects(); o++)
        {
            btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[o];
            btRigidBody* body = btRigidBody::upcast(obj);

            if (body && body->getMotionState())
            {                
                btTransform tr = body->getWorldTransform();
                btVector3 location = tr.getOrigin();
                btQuaternion rotation = tr.getRotation();

                if( o < locations_prev.size() )
                {
                    double dist_loc = compute_dist(location, locations_prev[o]);
                    double dist_rot = compute_dist(rotation, rotations_prev[o]);

                    //cout << "dist loc: "<< dist_loc << ", dist rot: " << dist_rot << endl;
                    dist_local += (dist_loc + dist_rot);
                    locations_prev[o] = location;
                    rotations_prev[o] = rotation;
                }
                else
                {
                    locations_init.push_back(location);
                    locations_prev.push_back(location);
                    rotations_init.push_back(rotation);
                    rotations_prev.push_back(rotation);
                }
            }            
        }

        dist_sum += dist_local;
        StepSimulation(0.1);

        //if( iter>0 && dist_local < eps ) break;        

        iter++;
    } while(iter < 100) ;
    
    return dist_sum;
}

static b3MouseMoveCallback prevMouseMoveCallback = 0;
static void OnMouseMove( float x, float y)
{
    bool handled = false; 
    //handled = example->mouseMoveCallback(x,y);
    if (!handled)
    {
        if (prevMouseMoveCallback)
            prevMouseMoveCallback (x,y);
    }
}

static b3MouseButtonCallback prevMouseButtonCallback  = 0;
static void OnMouseDown(int button, int state, float x, float y) 
{
    bool handled = false;
    //handled = example->mouseButtonCallback(button, state, x,y); 
    if (!handled)
    {
        if (prevMouseButtonCallback )
            prevMouseButtonCallback (button,state,x,y);
    }
}

static double colors_random[][4] = 
{
    {100, 100, 100, 255},
    {230,  25,  75, 255},
    { 60, 180,  75, 255},
    {255, 225,  25, 255},
    {  0, 130, 200, 255},
    {245, 130,  48, 255},
    {145,  30, 180, 255},
    { 70, 240, 240, 255},
    {240,  50, 230, 255},
    {210, 245,  60, 255},
    {250, 190, 190, 255},
    {  0, 128, 128, 255},
    {230, 190, 255, 255},
    {170, 110,  40, 255},
    {255, 250, 200, 255},
    {128,   0,   0, 255},
    {170, 255, 195, 255},
    {128, 128,   0, 255},
    {255, 215, 180, 255},
    {  0,   0, 128, 255},
    {128, 128, 128, 255},
    {255, 255,  25, 255}
};

vector<btVector4> MyOpenGLGuiHelper::colors;

void MyOpenGLGuiHelper::autogenerateGraphicsObjects(btDiscreteDynamicsWorld* rbWorld)
{
    btAlignedObjectArray<btCollisionObject*> sortedObjects;
    sortedObjects.reserve(rbWorld->getNumCollisionObjects());
    for (int i=0;i<rbWorld->getNumCollisionObjects();i++)
    {
        btCollisionObject* colObj = rbWorld->getCollisionObjectArray()[i];  
        createCollisionShapeGraphicsObject(colObj->getCollisionShape());

        btVector4 color;
        if( i > colors.size() )
        {
            double* clr
             = colors_random[ (i-colors.size())
                               % (sizeof(colors_random)/(sizeof(double)*4)) ];
            color = btVector4(clr[0]/255.,clr[1]/255.,clr[2]/255.,clr[3]/255.);
        }
        else
        {
            color = colors[i];
        }

        createCollisionObjectGraphicsObject(colObj,color);
    }
}

BulletSimulationGui::BulletSimulationGui() : BulletSimulation()
{
    m_gui = NULL;
    m_app = NULL;

    InitWorld();

    b_init_gui = false;
}

BulletSimulationGui::~BulletSimulationGui()
{
    ExitWorld();
}

void BulletSimulationGui::AddColor(btVector4 color)
{
    MyOpenGLGuiHelper::colors.push_back(color);
}

void BulletSimulationGui::ResetCamera( float dist, float pitch, float yaw, 
                                       float x, float y, float z           )
{
    m_gui->resetCamera(dist,pitch,yaw,x,y,z);
}

void BulletSimulationGui::SpinInit()
{    
    SpinExit();

    m_app = new SimpleOpenGL3App("Bullet Standalone Example",1024,768,true);
    
    prevMouseButtonCallback = m_app->m_window->getMouseButtonCallback();
    prevMouseMoveCallback = m_app->m_window->getMouseMoveCallback();
    m_app->m_window->setMouseButtonCallback((b3MouseButtonCallback)OnMouseDown);
    m_app->m_window->setMouseMoveCallback((b3MouseMoveCallback)OnMouseMove);

    m_gui = new MyOpenGLGuiHelper(m_app,false);
    m_gui->setUpAxis(2);
    m_gui->createPhysicsDebugDrawer(m_dynamicsWorld);
    if (m_dynamicsWorld->getDebugDrawer())
        m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawWireframe+btIDebugDraw::DBG_DrawContactPoints);
    m_gui->autogenerateGraphicsObjects(m_dynamicsWorld);
    //ResetCamera();

    b_init_gui = true;
}

void BulletSimulationGui::SpinExit()
{
    for (int i=0;i<m_dynamicsWorld->getNumCollisionObjects();i++)
    {        
        btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
        obj->getCollisionShape()->setUserIndex(-1);
        obj->setUserIndex(-1);
    }    
#if 1
    if(m_gui)
    {
        delete m_gui;
        m_gui = NULL;
    } 
    if(m_app)
    {
        delete m_app;
        m_app = NULL;        
    } 
#else

#endif
    b_init_gui = false;
}

void BulletSimulationGui::SpinOnce(float duration)
{
    if( !b_init_gui ) SpinInit();

    StepSimulation(duration);
}

void BulletSimulationGui::Spin(float speed)
{
    if( !b_init_gui ) SpinInit();

    b3Clock clock;    
    do
    {
        btScalar dtSec = btScalar(clock.getTimeInSeconds());
        dtSec *= speed;
        if (dtSec<0.1) dtSec = 0.1;

        StepSimulation(dtSec);

        clock.reset();
    } while (!m_app->m_window->requestedExit());
}

void BulletSimulationGui::StepSimulation(float duration)
{
    if( m_app )
    {
        m_app->m_instancingRenderer->init();
        m_app->m_instancingRenderer->updateCamera(m_app->getUpAxis());
    }

    BulletSimulation::StepSimulation(duration);
    
    if( m_gui )
    {
        m_gui->syncPhysicsToGraphics(m_dynamicsWorld);
        m_gui->render(m_dynamicsWorld);
    }

    if( m_app )
    {
#if 0
        DrawGridData dg;
        dg.upAxis = m_app->getUpAxis();
        m_app->drawGrid(dg);
#endif     
        m_app->swapBuffer();
    }
}