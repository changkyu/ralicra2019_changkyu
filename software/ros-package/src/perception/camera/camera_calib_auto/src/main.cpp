#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>

#include "TaskPlanner.hpp"

using namespace std;
using namespace pcl;

PointXYZRGB point;
string ROS_TOPIC_POINT;
pcl::visualization::PCLVisualizer viewer("Segmentation Result");
PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
string outdir;

bool running = true;
bool go_to_next = false;
bool compute = false;
bool refresh = false;
int idx=0;

void Callback_point(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    static bool busy=false;
    if( busy ) return;
    busy=true;

    fromROSMsg<PointXYZRGB>(*msg, *cloud);
    vector<int> index;
    removeNaNFromPointCloud(*cloud,*cloud,index);
    if(cloud->size() > 0)
    {        
        viewer.updatePointCloud(cloud);

        viewer.removeAllShapes();
        PointXYZRGB pt_x1 ,pt_x2, pt_y1, pt_y2, pt_z1, pt_z2;
        pt_x1 = point; pt_x2 = point;
        pt_y1 = point; pt_y2 = point;
        pt_z1 = point; pt_z2 = point;
        pt_x1.x = point.x - 0.01; pt_x2.x = point.x + 0.01;
        pt_y1.y = point.y - 0.01; pt_y2.y = point.y + 0.01;
        pt_z1.z = point.z - 0.01; pt_z2.z = point.z + 0.01;
        viewer.addLine(pt_x1, pt_x2, 0,0,1, "pt_x" );
        viewer.addLine(pt_y1, pt_y2, 0,0,1, "pt_y" );
        viewer.addLine(pt_z1, pt_z2, 0,0,1, "pt_z" );

        viewer.spinOnce(10);        
    }
    busy=false;
}

static
void Callback_viewer(const visualization::KeyboardEvent &event, void* viewer_void)
{
    visualization::PCLVisualizer *viewer
     = static_cast<visualization::PCLVisualizer *> (viewer_void);
    if (event.keyDown())
    {
        if( event.getKeySym()=="Escape" )
        {
            running = false;            
        }
        else if( event.getKeySym()=="space")
        {
            go_to_next = true;
        }
        else if( event.getKeySym()=="Return" )
        {
            compute = true;
        }        
        else if( event.getKeySym()=="F5" )
        {            
            refresh = true;
        }
    }
}

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
    if (event.getPointIndex () == -1)
    {
      return;
    }

    point = cloud->points[event.getPointIndex()];
    
    viewer.removeAllShapes();
    PointXYZRGB pt_x1 ,pt_x2, pt_y1, pt_y2, pt_z1, pt_z2;
    pt_x1 = point; pt_x2 = point;
    pt_y1 = point; pt_y2 = point;
    pt_z1 = point; pt_z2 = point;
    pt_x1.x = point.x - 0.01; pt_x2.x = point.x + 0.01;
    pt_y1.y = point.y - 0.01; pt_y2.y = point.y + 0.01;
    pt_z1.z = point.z - 0.01; pt_z2.z = point.z + 0.01;

    viewer.addLine(pt_x1, pt_x2, 0,0,1, "pt_x" );
    viewer.addLine(pt_y1, pt_y2, 0,0,1, "pt_y" );
    viewer.addLine(pt_z1, pt_z2, 0,0,1, "pt_z" );

    ROS_INFO_STREAM("select point: " << point);
}

static
void ParseParam(ros::NodeHandle nh)
{
    nh.param<string>("/camera_calib_auto/subtopic/pointcloud",ROS_TOPIC_POINT,"/pointcloud");
    nh.param<string>("/camera_calib_auto/outdir",outdir,getenv("HOME"));
}

int main( int argc, char** argv )
{
    ros::init(argc,argv,"camera_calib");
    ros::NodeHandle nh;
    ParseParam(nh);

    ros::Subscriber sub_point = nh.subscribe(ROS_TOPIC_POINT,10,Callback_point);
    
    viewer.setPosition(0,0);
    viewer.setSize(600,480);
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.addPointCloud(cloud);
    viewer.registerKeyboardCallback(Callback_viewer, (void*)&viewer);
    viewer.registerPointPickingCallback (pp_callback, (void*)&viewer); 

    TaskPlanner task_planner;

    while( ros::ok() && running )
    {
        ros::spinOnce();
        viewer.spinOnce(3);

        if( go_to_next )
        {
            stringstream ss;
            ss << outdir << "/ee_pose_" << idx << ".txt";
            ROS_INFO_STREAM("write to " << ss.str());

            ofstream f_ee(ss.str());
            geometry_msgs::PoseStamped p = task_planner.get_current_pose();
            f_ee << p.pose.position.x << endl;
            f_ee << p.pose.position.y << endl;
            f_ee << p.pose.position.z << endl;
            f_ee << p.pose.orientation.x << endl;
            f_ee << p.pose.orientation.y << endl;
            f_ee << p.pose.orientation.z << endl;
            f_ee << p.pose.orientation.w << endl;
            f_ee.close();
            ROS_INFO_STREAM("done");

            stringstream ss_cam;
            ss_cam << outdir << "/cam_pose_" << idx << ".txt";
            ROS_INFO_STREAM("write to " << ss_cam.str());

            ofstream f_cam(ss_cam.str());
            f_cam << point.x << endl;
            f_cam << point.y << endl;
            f_cam << point.z << endl;
            f_cam.close();
            ROS_INFO_STREAM("done");

            idx++;
            go_to_next = false;
        }
        if( refresh )
        {
            PointXYZRGB point_new;
            float dist_min = INFINITY;            
            for( size_t p=0; p<cloud->size(); p++ )
            {
                if( cloud->points[p].z < 0.01 ) continue;

                int dist_r = point.r - cloud->points[p].r;
                int dist_g = point.g - cloud->points[p].g;
                int dist_b = point.b - cloud->points[p].b;
                float dist = sqrt(dist_r*dist_r+dist_g*dist_g+dist_b*dist_b);
                if( dist < dist_min)
                {
                    dist_min = dist;
                    point_new = cloud->points[p];
                }
            }
            point = point_new;
            ROS_INFO_STREAM("point_new: " << point);

            viewer.removeAllShapes();
            PointXYZRGB pt_x1 ,pt_x2, pt_y1, pt_y2, pt_z1, pt_z2;
            pt_x1 = point; pt_x2 = point;
            pt_y1 = point; pt_y2 = point;
            pt_z1 = point; pt_z2 = point;
            pt_x1.x = point.x - 0.01; pt_x2.x = point.x + 0.01;
            pt_y1.y = point.y - 0.01; pt_y2.y = point.y + 0.01;
            pt_z1.z = point.z - 0.01; pt_z2.z = point.z + 0.01;

            viewer.addLine(pt_x1, pt_x2, 0,0,1, "pt_x" );
            viewer.addLine(pt_y1, pt_y2, 0,0,1, "pt_y" );
            viewer.addLine(pt_z1, pt_z2, 0,0,1, "pt_z" );

            refresh = false;
        }
    }
    ros::shutdown();

    return 0;
}
