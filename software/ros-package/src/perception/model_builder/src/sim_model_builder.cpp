
#include "model_builder/sim_model_builder.hpp"
#include "bullet_simulation/bullet_simulation.hpp"

using namespace std;
using namespace pcl;

static
bool Compare(const SimModelBuilder::VisConvexHull &first,
             const SimModelBuilder::VisConvexHull &second)
{   
    if( first.cz == second.cz )
    {
        if( first.cy == second.cy ) return first.cx > second.cx;        
        else                        return first.cy > second.cy;
    }
    else
    {
        return first.cz > second.cz;
    }
}

SimModelBuilder::SimModelBuilder()
{

}

SimModelBuilder::~SimModelBuilder()
{

}

void SimModelBuilder::AddInputVisModel(VisConvexHull &chull)
{
    _chulls.push_back(chull);
}

void SimModelBuilder::AddStaticVisModel(Plane &plane)
{
    _planes_bg.push_back(plane);
}

void SimModelBuilder::AddInputVisModels(vector<VisConvexHull> &chulls)
{
    for( size_t c=0; c<chulls.size(); c++ ) _chulls.push_back(chulls[c]);
}

void SimModelBuilder::AddStaticVisModels(std::vector<Plane>& planes)
{
    for( size_t p=0; p<planes.size(); p++ ) _planes_bg.push_back(planes[p]);    
}

float SimModelBuilder::TestVisModels()
{
    float dist =0;
    BulletSimulationGui sim;
    //sim.ResetCamera( 0.07,270,50, 0.5,0,0 );
    sim.ResetCamera( 0.07,180,0, 0.5,0.5,0 );
    
    size_t n_planes = _planes_bg.size();    
    for( size_t p=0; p<n_planes; p++ )
    {        
        sim.AddPlaneShape( _planes_bg[p] );
        sim.AddColor(btVector4(0.3,0.3,0.3,1));
    }

    size_t n_objs = _chulls.size();
    for( size_t o=0; o<n_objs; o++ )
    {  
        sim.AddConvexHullShape(*_chulls[o].cloud_hull, 
          btVector3( _chulls[o].cx, _chulls[o].cy, _chulls[o].cz ));

    }

#if 1
    int dummy;
    do
    {
        sim.SpinOnce();    
        std::cin >> dummy;
    }while( dummy );

#else
    sim.Spin(0.1);
#endif

    return dist;
}

float SimModelBuilder::TestVisModelsGui()
{
    float dist =0;
    BulletSimulationGui sim;
    //sim.ResetCamera( 0.07,270,50, 0.5,0,0 );
    sim.ResetCamera( 0.07,180,0, 0.5,0.5,0 );
    
    size_t n_planes = _planes_bg.size();    
    for( size_t p=0; p<n_planes; p++ )
    {        
        sim.AddPlaneShape( _planes_bg[p] );
        sim.AddColor(btVector4(0.3,0.3,0.3,1));
    }

    size_t n_objs = _chulls.size();
    for( size_t o=0; o<n_objs; o++ )
    {  
        sim.AddConvexHullShape(*_chulls[o].cloud_hull, 
          btVector3( _chulls[o].cx, _chulls[o].cy, _chulls[o].cz ));

    }

#if 1
    int dummy;
    do
    {
        sim.SpinOnce();    
        std::cin >> dummy;
    }while( dummy );

#else
    sim.Spin(0.1);
#endif

    return dist;
}

float SimModelBuilder::TestVisModelsTop2Bottom()
{
    float dist=0;

    BulletSimulationGui sim;
    //sim.ResetCamera( 0.07,270,50, 0.5,0,0 );
    sim.ResetCamera( 0.10,180,0, 0.5,0.5,0 );
    size_t n_planes = _planes_bg.size();    
    for( size_t p=0; p<n_planes; p++ )
    {        
        sim.AddPlaneShape( _planes_bg[p] );
    }

    sort( _chulls.begin(), _chulls.end(), Compare);

    size_t n_objs = _chulls.size();
    for( size_t o=0; o<n_objs; o++ )
    {        
        PointCloud<PointXYZRGB> cloud_others;        
        for( size_t o2=0; o2<n_objs; o2++ )
        {
            if( o2 != o )
            {                
                for( size_t p=0; p<_chulls[o2].cloud_hull->size(); p++ )
                {
                    PointXYZRGB pt;
                    pt.x = (*_chulls[o2].cloud_hull)[p].x + _chulls[o2].cx;
                    pt.y = (*_chulls[o2].cloud_hull)[p].y + _chulls[o2].cy;
                    pt.z = (*_chulls[o2].cloud_hull)[p].z + _chulls[o2].cz;
                    cloud_others.push_back(pt);
                }
            }
        }

        double cx=0, cy=0, cz=0;
        for( size_t p=0; p<cloud_others.size(); p++ )
        {
            cx += cloud_others[p].x;
            cy += cloud_others[p].y;
            cz += cloud_others[p].z;
        }
        cx /= cloud_others.size();
        cy /= cloud_others.size();
        cz /= cloud_others.size();
        for( size_t p=0; p<cloud_others.size(); p++ )
        {
            cloud_others[p].x -= cx;
            cloud_others[p].y -= cy;
            cloud_others[p].z -= cz;
        }

        sim.AddConvexHullShape(*_chulls[o].cloud_hull, 
          btVector3( _chulls[o].cx, _chulls[o].cy, _chulls[o].cz ));
        //sim.AddConvexHullShape(cloud_others, btVector3(cx,cy,cz));
        
#if 1
        //sim.Spin(0.1);
#else        
        do
        {
            sim.SpinOnce();
            cout << "Press Any key to continue or Press 'q' to quit ..." << endl;            
        }while( getchar() != 'q' );
#endif
    }

#if 1
    /*
    int dummy;
    do
    {
        sim.SpinOnce();    
        std::cin >> dummy;
    }while( dummy );
    */

    sim.Spin(0.1);
#else
    size_t n_objs = _chulls.size();    
    for( size_t o=0; o<n_objs; o++ )
    {        
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        pcl::fromPCLPointCloud2(_chulls[o].polymesh_hull->cloud, cloud);
        
        example.AddConvexHullShape(cloud, btVector3(_chulls[o].cx,
                                                    _chulls[o].cy,
                                                    _chulls[o].cz));
    }
    
    example.Spin(0.1);    
    /*
    int dummy;
    do
    {
        example.SpinOnce();    
        std::cin >> dummy;
    }while( dummy );
    */
#endif
    return dist;
}