#include <iostream>
#include <fstream>
#include <map>
#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

#include <omp.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/surface/vtk_smoothing/vtk_utils.h>

#include <vtkMassProperties.h>

#include "bullet_simulation/bullet_simulation.hpp"
#include "bullet_simulation/conversions.hpp"

#include "model_builder/vis_model_builder.hpp"

#include "utils/utils.hpp"

#include <shape_msgs/Plane.h>
#include <btBulletDynamicsCommon.h>

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;

Eigen::Matrix4f tf_cam2world;

vector<float> camera_K;
vector<float> camera_RT;
vector<float> workspace;
vector<float> groundplane;
vector<float> box;
string param_seg;

typedef struct GT
{
    string name;
    double tx;
    double ty;
    double tz;
    double qw;
    double qx;
    double qy;
    double qz;
} GT;

void toTransformationMatrix(Eigen::Matrix4f& camPose, struct GT &gt)
{
    camPose(0,3) = gt.tx;
    camPose(1,3) = gt.ty;
    camPose(2,3) = gt.tz;
    camPose(3,3) = 1;

    Eigen::Quaternionf q;
    q.w() = gt.qw;
    q.x() = gt.qx;
    q.y() = gt.qy;
    q.z() = gt.qz;
    Eigen::Matrix3f rotMat;
    rotMat = q.toRotationMatrix();

    for(int ii = 0;ii < 3; ii++)
        for(int jj=0; jj < 3; jj++)
        {
            camPose(ii,jj) = rotMat(ii,jj);
        }        
}

void ParseParam(ros::NodeHandle nh)
{
    nh.getParam("/segmentation/camera_info/intrinsic", camera_K);
    nh.getParam("/segmentation/camera_info/extrinsic", camera_RT);

    for( int i=0; i<16; i++ ) tf_cam2world(i/4,i%4) = camera_RT[i];

    cout << tf_cam2world << endl;
}

bool my_compare( VisModelBuilder::VisConvexHull* vis1, 
                 VisModelBuilder::VisConvexHull* vis2  )
{
    return vis1->likelihood > vis2->likelihood;
}

void readGrountTruth(char* fp_gt, vector<GT> &gts)
{
    ifstream if_gt(fp_gt);    
    string line;
    while( getline(if_gt,line) )
    {
        if( line.compare("")==0 ) break;
        istringstream iss(line);

        struct GT gt;        
        iss >> gt.name;
        iss >> gt.tx;
        iss >> gt.ty;
        iss >> gt.tz;
        iss >> gt.qw;
        iss >> gt.qx;
        iss >> gt.qy;
        iss >> gt.qz;
    
        cout << gt.name << ": ";
        cout << gt.tx << ", ";
        cout << gt.ty << ", ";
        cout << gt.tz << ", ";
        cout << gt.qw << ", ";
        cout << gt.qx << ", ";
        cout << gt.qy << ", ";
        cout << gt.qz << endl;

        gts.push_back(gt);
    }
    if_gt.close();
}

int main(int argc, char* argv[])
{
    omp_set_num_threads(omp_get_max_threads());

    ros::init(argc,argv,"experiment1");
    ros::NodeHandle nh;
    
    string dp_image;
    string fmt_image;
    string prefix;
    vector<int> range_index;
    bool stepbystep;
    int num_of_try;
    bool vis_pcl;
    string dp_save;
    int method;
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("inputdir,i",   po::value<string>(&dp_image),
                         "input image director")        
        ("prefix,p",     po::value<string>(&prefix)
                         ->default_value(""),
                         "input image filename format")
        ("indexrange,r", po::value<vector<int> >(&range_index)->multitoken(),
                         "input image index range (0,n)")
        ("bullet,b",       po::value<bool>(&stepbystep)->default_value(false),
                         "show every simulation step")
        ("save,s",       po::value<string>(&dp_save)->default_value(""),
                         "the directory to save the results")
        ("num_of_try,n", po::value<int>(&num_of_try)->default_value(100),
                         "# of hypothesis")
        ("vis,v",        po::value<bool>(&vis_pcl)->default_value(false),
                         "visualize model")
        ("method,m",     po::value<int>(&method)->default_value(1),
                         "method=0 all method=1 order")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if( vm.count("help")        ||
        dp_image.compare("")==0 || 
        range_index.size()!=2      ) 
    {
        cout << dp_image << endl;
        cout << prefix << endl;
        cout << range_index.size() << endl;
        cout << desc << "\n";
        return 0;
    }

    if( prefix.compare("") ) prefix += ".";
    
    // Read inputs
    const int n_images = range_index[1] - range_index[0] + 1;
    vector<Mat> images(n_images);
    vector<Mat> depths(n_images);
    vector<PointCloud<PointXYZRGB>::Ptr> clouds(n_images);
    for( int i=0; i<n_images; i++ )
    {        
        char fp_image[256];
        sprintf(fp_image,(dp_image + "/" + "%s%06d.color.png").c_str(),
                prefix.c_str(),i+range_index[0]);
        cout << fp_image << endl;
        images[i] = imread(fp_image);

        char fp_depth[256];
        sprintf(fp_depth,(dp_image + "/" + "%s%06d.depth.png").c_str(),
                prefix.c_str(),i+range_index[0]);
        cout << fp_depth << endl;
        depths[i] = imread(fp_depth,CV_LOAD_IMAGE_ANYDEPTH);

        char fp_cloud[256];
        sprintf(fp_cloud,(dp_image + "/" + "%s%06d.cloud.pcd").c_str(),
                prefix.c_str(),i+range_index[0]);
        
        clouds[i] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        if (io::loadPCDFile<PointXYZRGB> (fp_cloud, *clouds[i]) == -1)
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
            return (-1);
        }
        cout << fp_cloud << endl;
    }

    // Read ground truth info
    char fp_gt[256];
    sprintf(fp_gt,(dp_image + "/" + "%s%06d.gt.txt").c_str(), prefix.c_str(),range_index[0]);    
    vector<struct GT> gts;
    readGrountTruth(fp_gt, gts);

    PolygonMesh meshs_gt[gts.size()];
    PointCloud<PointXYZRGB>::Ptr clouds_gt[gts.size()];
    for( size_t g=0; g<gts.size(); g++ )
    {
        sprintf(fp_gt,"/home/cs1080/projects/dataset/YCB/ycb/%s/tsdf/textured.obj",
                       gts[g].name.c_str());        
        string tmp(fp_gt);        
        if( pcl::io::loadPolygonFileOBJ(tmp, meshs_gt[g]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_gt);
            return (-1);
        }

        clouds_gt[g] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        sprintf(fp_gt,"/home/cs1080/projects/dataset/YCB/ycb/%s/clouds/merged_cloud.ply",
                       gts[g].name.c_str());        
        string tmp2(fp_gt);
        pcl::io::loadPLYFile(tmp2, *clouds_gt[g]);

        VoxelGrid< PointXYZRGB > sor;        
        sor.setInputCloud (clouds_gt[g]);
        sor.setLeafSize (0.0025f, 0.0025f, 0.0025f);
        sor.filter (*clouds_gt[g]);

        Eigen::Matrix4f tf;
        toTransformationMatrix(tf, gts[g]);
        transformPointCloud(*clouds_gt[g], *clouds_gt[g], tf);

        PointCloud<PointXYZ> cloud;
        fromPCLPointCloud2(meshs_gt[g].cloud, cloud);
        transformPointCloud(cloud, cloud, tf);
        toPCLPointCloud2(cloud, meshs_gt[g].cloud); 
    }

/*
    {
        stringstream ss;
        ss << "num_of_objects=" << gts.size();
        param_seg = ss.str();
    }
*/
    // Read scene info
    {
        char fp_sceneinfo[256];
        sprintf(fp_sceneinfo, (dp_image + "/%s").c_str(), "scene_info.yaml");
        try
        {
            YAML::Node lconf = YAML::LoadFile(fp_sceneinfo);

            camera_K  = lconf["intrinsic"].as<vector<float> >();
            camera_RT = lconf["extrinsic"].as<vector<float> >();            
            if( lconf["groundplane"] )
                groundplane = lconf["groundplane"].as<vector<float> >();
            if( lconf["box"] )
            {
                box = lconf["box"].as<vector<float> >();

                if( param_seg.compare("")!=0 ) param_seg += ",";
                param_seg += "remove_background=0";
            }
            else
            {
                if( param_seg.compare("")!=0 ) param_seg += ",";
                param_seg += "remove_background=1";
            }
            if( lconf["workspace"] )
            {
                workspace = lconf["workspace"].as<vector<float> >();
                stringstream ss;
                ss << "workspace=" << workspace[0] << " " << 
                                      workspace[1] << " " << 
                                      workspace[2] << " " << 
                                      workspace[3] << " " << 
                                      workspace[4] << " " << 
                                      workspace[5];

                if( param_seg.compare("")!=0 ) param_seg += ",";
                param_seg += ss.str();
            }

            for( int i=0; i<16; i++ ) tf_cam2world(i/4,i%4) = camera_RT[i];
            cout << tf_cam2world << endl;
        }
        catch(Exception &e)
        {
            ROS_ERROR_STREAM("Can't find such a cam_info file: " << fp_sceneinfo);
            ParseParam(nh);
        }
    }
cout << "PARAM: " << param_seg << endl;

    // Create Visual model
    VisModelBuilder vis( &nh, camera_K, camera_RT, 
                         ("name_2dseg=quickshift" + ("," + param_seg)), 
                         images[0].cols, images[0].rows);
    
    // Add ground / workspace info
    if( groundplane.size() > 0 )
    {
        shape_msgs::Plane gdplane;
        gdplane.coef[0] =  groundplane[0];
        gdplane.coef[1] =  groundplane[1];
        gdplane.coef[2] =  groundplane[2];
        gdplane.coef[3] =  groundplane[3];
        vis.AddGroundPlane(gdplane);
    }    
    if( box.size() > 0 )
    {
        vis.AddStaticBoxOpened(box);
    }

    const int end=images.size()-1;
    if( images.size() > 1 )
    {
        for( size_t i=0; i<end; i++ )
        {        
            ROS_INFO_STREAM( i << " - update");
            if( 0<i && i<end )
            {
                vis.Update(images[i],depths[i]);
            }
            else
            {
                vis.Update(images[i],depths[i],clouds[i]);
            }
            ROS_INFO_STREAM( i << " - update - Done");
        }
    }
    else
    {
        ROS_INFO_STREAM( "0 = update");
        vis.Update(images[0],depths[0],clouds[0]);            
        ROS_INFO_STREAM( "update - Done");
    }
    

    //int num_of_hypotheses = (int)(exp(log(num_of_try)/gts.size()));
    //if( num_of_hypotheses < 3 ) num_of_hypotheses = 3;    
    int num_of_hypotheses = 16;

for( size_t t=0; t<1; t++ )
{
    size_t n_objs = vis.NumOfModels();
    VisModelBuilder::VisConvexHull vischulls[n_objs][NUM_OF_HYPS];
    VisModelBuilder::VisConvexHull chulls_grv[n_objs];
    VisModelBuilder::VisConvexHull chulls_two[n_objs];
    VisModelBuilder::VisConvexHull* chulls_max[n_objs];
    VisModelBuilder::VisConvexHull* chulls_top5[n_objs][5];
    vector<VisModelBuilder::VisConvexHull*> most_stable;

    vector<vector<pair<float,VisModelBuilder::VisConvexHull*> > > probs;
if( method==0 )
{
    if( end > 0 ) vis.Update(images[end],depths[end],clouds[end]);
        // Update the inputs
    const int end=images.size()-1;    
    for( size_t i=0; i<=end; i++ )
    {        
        ROS_INFO_STREAM( i << " - update");
        if( 0<i && i<end )
        {
            vis.Update(images[i],depths[i]);
        }
        else
        {
            vis.Update(images[i],depths[i],clouds[i]);
        }
        ROS_INFO_STREAM( i << " - update - Done");
    }

    num_of_try = 100*num_of_hypotheses*(n_objs+1)*n_objs/2.0;

    cout << "# of hypotheses to generate for each object: " << num_of_hypotheses << endl;
    cout << "# of simulation: " << num_of_try << endl;

    // Generate the models    
    ROS_INFO_STREAM("GenerateModels");    
    vector<VisModelBuilder::VisConvexHull*> chulls_rnd[n_objs];
    for( size_t o=0; o<n_objs; o++ )
    {
        chulls_rnd[o].resize(num_of_hypotheses);
        vis.GenerateGravityModel(o,chulls_grv[o]);
        vis.GenerateTwoFacesModel(o,chulls_two[o]);

        size_t h=0;
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateGravityModel(o, *chulls_rnd[o][h]); h++;
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_MINIMUM   | VisModelBuilder::NO_NEWFACE); h++;        
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_MAXIMUM_H | VisModelBuilder::NO_NEWFACE); h++;
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_MAXIMUM_V | VisModelBuilder::NO_NEWFACE); h++;
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_MAXIMUM_H | VisModelBuilder::GEN_NEWFACE); h++;
        chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
        vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_MAXIMUM_V | VisModelBuilder::GEN_NEWFACE); h++;
        for( ; h<num_of_hypotheses; h++ )
        {            
            chulls_rnd[o][h] = new VisModelBuilder::VisConvexHull;
            int flag = (rand() % 10)==0? VisModelBuilder::GEN_NEWFACE : VisModelBuilder::NO_NEWFACE;
            vis.GenerateModel(o, *chulls_rnd[o][h], VisModelBuilder::GEN_RANDOM | flag);
        }

        sort(chulls_rnd[o].begin(), chulls_rnd[o].end(), my_compare);
        ROS_INFO_STREAM( "GenerateModels (" << o+1 << "/" << n_objs << ")");
    }
    ROS_INFO_STREAM("GenerateModels - Done");

    VisModelBuilder::VisConvexHull* chulls_vislike_max[n_objs];
    VisModelBuilder::VisConvexHull* chulls_simlike_max[n_objs];    

    float vislike_max = -INFINITY;
    float simlike_max = -INFINITY;
    float score_max   = -INFINITY;

    int idx_score_max = -1;

    int idx_min = -1;
    float dist_min = INFINITY;

    float dist =0;    
    vector<shape_msgs::Plane> planes[num_of_try];
    double dists[num_of_try];

    #pragma omp parallel for
    for( size_t i=0; i<num_of_try; i++ )
    {    
        size_t n_models = vis.NumOfModels();

//        ROS_INFO_STREAM( i << " - Simulation");
        BulletSimulation sim;
        sim.InitWorld();

        vector<shape_msgs::Plane> &planes_bg = vis.GetBGPlanes();
        size_t n_planes = planes_bg.size();
        for( size_t p=0; p<n_planes; p++ )
        {   
            planes[i].push_back(planes_bg[p]);
            if( planes[i][p].coef[2] > 0.95 )
            {                
                sim.AddPlaneShape( planes[i][p] );

                sim.SetGravity(btVector3(-planes[i][p].coef[0],
                                         -planes[i][p].coef[1],
                                         -planes[i][p].coef[2]  ));
            } 
        }

        if( box.size() > 0 )
        {            
            sim.AddBucketShape(box,0);            
        }        

        VisModelBuilder::VisConvexHull* pchulls[n_objs];
        float vislike = 0;
        for( size_t o=0; o<n_objs; o++ )
        {
            VisModelBuilder::VisConvexHull* pvischull;
            int g_rnd;

            if(      i==0 )   g_rnd = 0;
            else if( i==1 )   g_rnd = 1;            
            else if( i < 20 ) g_rnd = rand() % 2;
            else              g_rnd = rand() % 3;

g_rnd = 3;

            if( g_rnd==0 )
            {                
                pvischull = &(chulls_grv[o]);                
            }
            else if( g_rnd==1 )
            {
                pvischull = &(chulls_two[o]);
            }
            else
            {
                /*
                int r_rnd;
                if( num_of_try > 2)
                {
                    r_rnd = rand() % (num_of_try/2);
                }
                else
                {
                    r_rnd = rand() % (num_of_try);
                }
                */
                int r_rnd = rand() % (num_of_hypotheses);
                pvischull = chulls_rnd[o][r_rnd];
            }
            vislike += (   pvischull->likelihood_face
                         - pvischull->penalty_var     );

            sim.AddConvexHullShape(*pvischull->cloud_hull, 
              btVector3( pvischull->cx, pvischull->cy, pvischull->cz ));

            pchulls[o] = pvischull;
        }
        vislike /= n_objs;

        if( vislike > vislike_max )
        {
            for( size_t o=0; o<n_objs; o++ )
            {
                chulls_vislike_max[o] = pchulls[o];
            }            
            vislike_max = vislike;
        }

        double dist = sim.SpinUntilStable();
        dists[i] = dist;
        
        if( dist < dist_min )
        {
            for( size_t o=0; o<n_objs; o++ )
            {
                chulls_simlike_max[o] = pchulls[o];
            } 
            idx_min = i;
            dist_min = dist;
        }
        double simlike = 1/(1+exp(4*(dist-4)));
//        double score = vislike * simlike;
        double score = -dist;
        if( score_max < score )
        {
            for( size_t o=0; o<n_objs; o++ )
            {
                chulls_max[o] = pchulls[o];
            } 
            score_max = score;
            idx_score_max = i;
        }

        sim.ExitWorld();
    }
}
else if( method == 1)
{
        // Update the inputs
    

    ROS_INFO_STREAM("GenerateModels");    
    for( size_t o=0; o<n_objs; o++ )
    {        
        vis.GenerateGravityModel(o,chulls_grv[o]);
        vis.GenerateTwoFacesModel(o,chulls_two[o]);        
    }
    vis.GenerateModels(vischulls);
    ROS_INFO_STREAM("GenerateModels - Done");

    for( size_t o=0; o<n_objs; o++ )
    {
        char fp_save[256];

        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
        
        vis.GetSegmentation(o, cloud);
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.pcd").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "seg", o);
cout << fp_save << endl;        
        io::savePCDFileBinary(fp_save, *cloud);
        
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.model.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "grv", o);
cout << fp_save << endl;        

        io::savePLYFile (fp_save, chulls_grv[o].polymesh);
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.model.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "two", o);
cout << fp_save << endl;        

        io::savePLYFile (fp_save, chulls_two[o].polymesh);
        
        for( size_t i=0; i<32; i++ )
        {
            sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.%d.model.ply").c_str(),
                    prefix.c_str(), range_index[0], range_index[1], "our", o, i);
cout << fp_save << endl;        
            
            io::savePLYFile (fp_save, vischulls[o][i].polymesh);
            cout << "save o=" << o << " i=" << i << endl;
        }
    }

    ROS_INFO_STREAM("Sampling");
    vector<pair<float,vector<VisModelBuilder::VisConvexHull*> > > res;    
    vector<VisModelBuilder::VisConvexHull*> best;
    vis.Sampling(vischulls, res, probs, best, most_stable);
    ROS_INFO_STREAM("Sampling - Done");

    if( images.size() > 1 && 0<end )
    {
        vis.Update(images[end],depths[end],clouds[end]);
        vis.Sampling2(probs, best, most_stable);
    }

    for( size_t o=0; o<n_objs; o++ )
    {
        chulls_max[o] = best[o];
        chulls_top5[o][0] = probs[o][0].second;
        chulls_top5[o][1] = probs[o][1].second;
        chulls_top5[o][2] = probs[o][2].second;
        chulls_top5[o][3] = probs[o][3].second;
        chulls_top5[o][4] = probs[o][4].second;
    }    

    /*
    visualization::PCLVisualizer viewer;     
    viewer.setWindowName("debug");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1, 1, 1);    
    for( size_t o=0; o<n_objs; o++ )
    {
        stringstream ss;
        ss << o;
        viewer.addPolygonMesh(chulls_max[o]->polymesh, "poly" + ss.str());
        cout << chulls_max[o] << endl;
    }
    viewer.spin();
    */
}

if( dp_save.compare("")!=0 )
{
    ROS_INFO_STREAM("Compute AoverU");

    PolygonMesh polymesh_gt[gts.size()];
    for( size_t g=0; g<gts.size(); g++ )
    {
        ConvexHull<PointXYZRGB> chull;
        chull.setInputCloud (clouds_gt[g]);
        chull.reconstruct (polymesh_gt[g]);    
    }

    double AoUs_grv[n_objs], 
           AoUs_two[n_objs],
           AoUs_our[n_objs][gts.size()],
           AoUs_our2[n_objs][5],
           AoUs_our3[n_objs];
    double AoU_grv[n_objs], AoU_two[n_objs], AoU_our[n_objs];

    cout << "[";
    for( size_t i=0; i<n_objs*gts.size(); i++ ) cout << " ";
    cout << "]" << endl;
    cout << "[";
    #pragma omp parallel for
    for( size_t i=0; i<n_objs*gts.size(); i++ )
    {    
        int o= i % n_objs;
        int g= i / n_objs;
                
        AoUs_our[o][g] = utils::IntersectionOverUnion(
                            chulls_max[o]->polymesh, polymesh_gt[g] );
        cout <<  "^" << std::flush;
    }
    cout << "]" << endl;
    
    int idxes_best[n_objs];
    #pragma omp parallel for
    for( size_t o=0; o<n_objs; o++ )
    {
        double AoU_best = 0;    
        for( size_t g=0; g<gts.size(); g++ )
        {
            if( AoU_best < AoUs_our[o][g] )
            {
                idxes_best[o] = g;
                AoU_best = AoUs_our[o][g];
            }
        }

        if( AoU_best==0 )
        {
            idxes_best[o] = -1;
            AoUs_grv[o] = 0;
            AoUs_two[o] = 0;
            if( method == 1 )
            {   
                for( size_t i=0; i<5; i++ )
                {
                    AoUs_our2[o][i] = 0;
                }                
                AoUs_our3[o] = 0;
            }                        
        }
        else
        {
            size_t g = idxes_best[o];
            AoUs_grv[o] = utils::IntersectionOverUnion(
                                chulls_grv[o].polymesh, polymesh_gt[g] );
            AoUs_two[o] = utils::IntersectionOverUnion(
                                chulls_two[o].polymesh, polymesh_gt[g] );

            if( method == 1 )
            {   
                for( size_t i=0; i<5; i++ )
                {
                    AoUs_our2[o][i] = probs[o][i].first * utils::IntersectionOverUnion(
                                     chulls_top5[o][i]->polymesh, polymesh_gt[g] );
                }
                
                AoUs_our3[o] = utils::IntersectionOverUnion(
                                most_stable[o]->polymesh, polymesh_gt[g] );            
            }
        }
    }

    int no_count = 0;
    double aver_grv=0, aver_two=0, aver_our=0, aver_our2=0, aver_our3=0;
    for( size_t o=0; o<n_objs; o++ )
    {
        int g = idxes_best[o];        
        if( g==-1 )
        { 
            no_count++;
            continue;
        }

        aver_grv += AoUs_grv[o];
        aver_two += AoUs_two[o];
        aver_our += AoUs_our[o][g];
        if( method == 1 )
        {
            aver_our2 += ( AoUs_our2[o][0] +
                           AoUs_our2[o][1] +
                           AoUs_our2[o][2] +
                           AoUs_our2[o][3] +
                           AoUs_our2[o][4]   );

            aver_our3 += AoUs_our3[o];
        }
    }
    aver_grv /= (n_objs-no_count);
    aver_two /= (n_objs-no_count);
    aver_our /= (n_objs-no_count);
    if( method == 1 )
    {
        aver_our2 /= (n_objs-no_count);
        aver_our3 /= (n_objs-no_count);
    }

    cout << "gravity: " << aver_grv << endl;
    cout << "twoface: " << aver_two << endl;
    cout << "ours:    " << aver_our << endl;
    if( method == 1 )
    {
        cout << "ours2:   " << aver_our2 << endl;
        cout << "ours3:   " << aver_our3 << endl;
    }
    //cout << score_max << endl;
    ROS_INFO_STREAM("Compute AoverU - Done");

    char fp_save[256];

    sprintf(fp_save,(dp_save + "/" + "%s%d-%d.result.txt").c_str(),
                     prefix.c_str(), range_index[0], range_index[1]);
    ofstream of(fp_save);
    of << "aver " << endl;
    of << "grv " << aver_grv << endl;
    of << "two " << aver_two << endl;
    of << "our " << aver_our << endl;
    of << "our2 " << aver_our2 << endl;
    of << "our3 " << aver_our3 << endl;

    for( size_t o=0; o<n_objs; o++ )
    {
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.result.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "grv", o);
        io::savePLYFile (fp_save, chulls_grv[o].polymesh);
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.result.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "two", o);
        io::savePLYFile (fp_save, chulls_two[o].polymesh);    
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.result.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "our", o);
        io::savePLYFile (fp_save, chulls_max[o]->polymesh);        

        if( method == 1 )
        {
            for( size_t i=0; i<5; i++ )
            {
                sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.%d.result.ply").c_str(),
                        prefix.c_str(), range_index[0], range_index[1], "our2", o, i);
                io::savePLYFile (fp_save, chulls_top5[o][0]->polymesh);
            }
            sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.result.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "our3", o);
            io::savePLYFile (fp_save, most_stable[o]->polymesh);        
        }

        of << "o=" << o << " gt=" << idxes_best[o] << endl;
        of << "grv " << AoUs_grv[o] << endl;
        of << "two " << AoUs_two[o] << endl;
        of << "our " << AoUs_our[o][idxes_best[o]] << endl;                
        if( method == 1 )
        {
            float norm = ( probs[o][0].first + 
                           probs[o][1].first + 
                           probs[o][2].first + 
                           probs[o][3].first + 
                           probs[o][4].first   );
            of << "our2 " << ( AoUs_our2[o][0] +
                               AoUs_our2[o][1] +
                               AoUs_our2[o][2] +
                               AoUs_our2[o][3] +
                               AoUs_our2[o][4] ) << " ";
            of << AoUs_our2[o][0] << " (" << probs[o][0].first << ") ";
            of << AoUs_our2[o][1] << " (" << probs[o][1].first << ") ";
            of << AoUs_our2[o][2] << " (" << probs[o][2].first << ") ";
            of << AoUs_our2[o][3] << " (" << probs[o][3].first << ") ";
            of << AoUs_our2[o][4] << " (" << probs[o][4].first << ")" << endl;

            of << "our3 " << AoUs_our3[o] << endl;
        }
    }
    for( size_t g=0; g<gts.size(); g++ )
    {
        sprintf(fp_save,(dp_save + "/" + "%s%d-%d.%s.%d.result.ply").c_str(),
                prefix.c_str(), range_index[0], range_index[1], "gt", g);
        io::savePLYFile (fp_save, meshs_gt[g]);
    }

    of.close();
}

if( vis_pcl )
{
    visualization::PCLVisualizer viewer; 
    int v1, v2, v3, v4, v5, v6, v7, v8, v9;
    viewer.createViewPort (0.00, 0.66, 0.33, 0.99, v1);
    viewer.createViewPort (0.33, 0.66, 0.66, 0.99, v2);
    viewer.createViewPort (0.66, 0.66, 0.99, 0.99, v3);
    viewer.createViewPort (0.00, 0.33, 0.33, 0.66, v4);
    viewer.createViewPort (0.33, 0.33, 0.66, 0.66, v5);
    viewer.createViewPort (0.66, 0.33, 0.99, 0.66, v6);
    viewer.createViewPort (0.00, 0.00, 0.33, 0.33, v7);
    viewer.createViewPort (0.33, 0.00, 0.66, 0.33, v8);
    viewer.createViewPort (0.66, 0.00, 0.99, 0.33, v9);
    viewer.setWindowName("ground truth");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1, 1, 1);
    PointCloud<PointXYZRGB>::Ptr cloud_world(new PointCloud<PointXYZRGB>);
    transformPointCloud(*clouds[0], *cloud_world, tf_cam2world);    
    viewer.addPointCloud(cloud_world, "cloud", v1);
    vis.DrawSegmentation(0,viewer,v2,v3);

    for( size_t g=0; g<gts.size(); g++ )
    {
        stringstream ss;
        ss << "gt" << g;
        //viewer.addTextureMesh(meshs_gt[g], ss.str(), v5);
        viewer.addPolygonMesh(meshs_gt[g], ss.str(), v8);
    }

    vector<PointCloud<PointXYZRGB>::Ptr> clouds_grv(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        Eigen::Matrix4f tf;
        tf << 1, 0, 0, chulls_grv[o].cx,
              0, 1, 0, chulls_grv[o].cy,
              0, 0, 1, chulls_grv[o].cz,
              0, 0, 0, 1;    
        clouds_grv[o] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        transformPointCloud(*chulls_grv[o].cloud, *clouds_grv[o], tf);
        transformPointCloud(*clouds_grv[o], *clouds_grv[o], chulls_grv[o].tf_cur2init);

        stringstream ss;
        ss << "gravity" << o;
        viewer.addPointCloud(clouds_grv[o], ss.str(), v4);
        viewer.addPolygonMesh(chulls_grv[o].polymesh, ss.str() + "poly", v4);
        viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 
                                                utils::colors_vis[o][0]/255.,
                                                utils::colors_vis[o][1]/255.,
                                                utils::colors_vis[o][2]/255.,
                                                ss.str() + "poly", v4);
    }

    vector<PointCloud<PointXYZRGB>::Ptr> clouds_two(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        Eigen::Matrix4f tf;
        tf << 1, 0, 0, chulls_two[o].cx,
              0, 1, 0, chulls_two[o].cy,
              0, 0, 1, chulls_two[o].cz,
              0, 0, 0, 1;    
        clouds_two[o] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        transformPointCloud(*chulls_two[o].cloud, *clouds_two[o], tf);
        transformPointCloud(*clouds_two[o], *clouds_two[o], chulls_two[o].tf_cur2init);
        
        stringstream ss;
        ss << "twofaces" << o;
        viewer.addPointCloud(clouds_two[o], ss.str(), v5);
        viewer.addPolygonMesh(chulls_two[o].polymesh, ss.str() + "poly", v5);
        viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 
                                                utils::colors_vis[o][0]/255.,
                                                utils::colors_vis[o][1]/255.,
                                                utils::colors_vis[o][2]/255.,
                                                ss.str() + "poly", v5);
    }

    vector<PointCloud<PointXYZRGB>::Ptr> clouds_max(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        Eigen::Matrix4f tf;
        tf << 1, 0, 0, chulls_max[o]->cx,
              0, 1, 0, chulls_max[o]->cy,
              0, 0, 1, chulls_max[o]->cz,
              0, 0, 0, 1;    
        clouds_max[o] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        transformPointCloud(*chulls_max[o]->cloud, *clouds_max[o], tf);
        transformPointCloud(*clouds_max[o], *clouds_max[o], chulls_max[o]->tf_cur2init);

        stringstream ss;
        ss << "best_model" << o;
        viewer.addPointCloud(clouds_max[o], ss.str(), v6);

        viewer.addPolygonMesh(chulls_max[o]->polymesh, ss.str() + "poly", v6);
        viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 
                                                utils::colors_vis[o][0]/255.,
                                                utils::colors_vis[o][1]/255.,
                                                utils::colors_vis[o][2]/255.,
                                                ss.str() + "poly", v6);
    }
    viewer.spin();
/*
    // Show the best
    int i=idx_min;
    BulletSimulationGui gui;
    for( size_t p=0; p<planes[i].size(); p++ )
    {   
        if( planes[i][p].coef[2] > 0.95 )
        {
            gui.AddPlaneShape( planes[i][p] );
            gui.AddColor(btVector4(0.3,0.3,0.3,1));
            gui.SetGravity(btVector3(-planes[i][p].coef[0]*10,
                                     -planes[i][p].coef[1]*10,
                                     -planes[i][p].coef[2]*10  ));
        } 
    }
    if( box.size() > 0 )
    {
        gui.AddBucketShape(box,0);
        gui.AddColor(btVector4(0.6,0.6,0.6,1));        
    }
    for( size_t o=0; o<n_objs; o++ )
    {  
        gui.AddConvexHullShape( *chulls_max[o]->cloud_hull, 
                              btVector3( chulls_max[o]->cx, 
                                         chulls_max[o]->cy, 
                                         chulls_max[o]->cz ));
    }

    gui.ResetCamera( 0.25,-90,45, 0.12,0,0 );
    if( stepbystep )
    {
        int dummy;
        do
        {
            gui.SpinOnce();    
            std::cin >> dummy;
        }while( dummy );
    }
    else
    {
        gui.Spin(0.1);
    }

    for( size_t o=0; o<n_objs; o++ )
    {  
        gui.AddConvexHullShape(*chulls_max[o]->cloud_hull, 
                             btVector3( chulls_max[o]->cx, 
                                        chulls_max[o]->cy, 
                                        chulls_max[o]->cz ));

    }
*/
}

}
}