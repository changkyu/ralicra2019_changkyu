#include <boost/program_options.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include "rl_msgs/tracking_image_srv.h"

#include "tracking/image_tracker.hpp"
#include "tracking/seg_tracker.hpp"
#include "tracking/colors.hpp"

#include "segmentation/quickshift/quickshift_wrapper.hpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

static vector<Vec3b> colors;
void drawLabels(Mat &input, Mat &dst)
{
    Mat markers;
    input.convertTo(markers,CV_32SC1);

    double minVal, maxVal;
    minMaxLoc(markers,&minVal,&maxVal);
    int n_colors = (int)maxVal;

    // Generate random colors
    for (size_t i = colors.size(); i < (n_colors+1); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors    
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            //int index = markers.at<int>(i,j);
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= n_colors)
            {
                dst.at<Vec3b>(i,j) = colors[index-1];                
            }
            else
            {
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);                
            }
        }
    }
}

int main(int argc, char* argv[])
{
    string dp_image;
    string fmt_image;
    vector<int> range_index;
    bool show_all;
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("inputdir,i",   po::value<string>(&dp_image),
                         "input image director")
        ("inputfmt,f",   po::value<string>(&fmt_image)
                         ->default_value("%06d.color.png"),
                         "input image filename format")
        ("indexrange,r", po::value<vector<int> >(&range_index)->multitoken(),
                         "input image index range (0,n)")
        ("allpoints,a", po::value<bool>(&show_all)->default_value(false),
                         "show all points")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if( vm.count("help")        ||
        dp_image.compare("")==0 || 
        range_index.size()!=2      ) 
    {
        cout << dp_image << endl;
        cout << fmt_image << endl;
        cout << range_index.size() << endl;
        cout << desc << "\n";
        return 0;
    }

    // ROS init
    ros::init(argc,argv,"example_image_tracker");
    ros::NodeHandle nh;      
    ros::ServiceClient clt_image_tracker
     = nh.serviceClient<rl_msgs::tracking_image_srv>("tracking/sift_image");

    rl_msgs::tracking_image_srv srv;
    const int n_images = range_index[1] - range_index[0] + 1;
    srv.request.images.resize(n_images);
    vector<Mat> images(n_images);
    for( int i=0; i<n_images; i++ )
    {        
        char fp_image[256];
        sprintf(fp_image,(dp_image + "/" + fmt_image).c_str(),i+range_index[0]);
        cout << fp_image << endl;
        images[i] = imread(fp_image);

        cv_bridge::CvImage msg_image;    
        msg_image.encoding = sensor_msgs::image_encodings::BGR8;
        msg_image.image    = images[i];
        msg_image.toImageMsg(srv.request.images[i]);
    }

    #define IMAGE_WIDTH  (600)
    #define IMAGE_HEIGHT (480)
    ImageTracker _imgTracker(IMAGE_WIDTH, IMAGE_HEIGHT);

    vector<vector<Point2f> > points_track;
    ROS_INFO_STREAM("Track");
    //_imgTracker.Track(images, points_track);
    for( size_t i=0; i<images.size(); i++ )
    {
        _imgTracker.AddImage(images[i]);
    }
    _imgTracker.Track(points_track);

    ROS_INFO_STREAM("Track - Done");

    ROS_INFO_STREAM("Segmentation");
    vector<Mat> images_seg(n_images);
    for( int i=0; i<n_images; i++ )
    {
        Segmentation_QuickShift(images[i], images_seg[i], 3, 10);
    }
    ROS_INFO_STREAM("Segmentation - Done");

    ROS_INFO_STREAM("Find Trajectories");
    vector<set<pair<size_t,size_t> > > traj;
    SegTrajectories(images_seg, points_track, traj );
    ROS_INFO_STREAM("Find Trajectories - Done");

    bool flag = true;
    int idx=0;
    while( flag )
    {
        Mat img_show;
        images[idx].copyTo(img_show);        
        if( idx>0 )
        {
            for( size_t p=0; p<points_track[idx].size(); p++ )
            {
                pair<size_t,size_t> vertex(idx,p);
                bool found = false;
                int label = 0;                
                for( label=0; label<traj.size(); label++ )
                {
                    if( traj[label].find(vertex) != traj[label].end() )
                    {
                        found = true;
                        break;
                    }
                }
                Scalar color = GetColor(label);
                stringstream ss;
                ss << label;

                Point2f &pt_prev = points_track[idx-1][p];
                Point2f &pt_curr = points_track[idx  ][p];
                 
                if( pt_prev.x >= 0 && pt_prev.y >= 0 &&
                    pt_curr.x >= 0 && pt_curr.y >= 0 && found )
                {
                    double dist = cv::norm(pt_prev-pt_curr);
                    if( dist > 5 || show_all)
                    {
                        putText(img_show, ss.str(), pt_curr, 
                                FONT_HERSHEY_PLAIN, 0.6, color,1);
                        circle(img_show, pt_curr, 1, color,2 );
                        line(img_show, pt_prev, pt_curr, color, 1);                
                    }
                }            
            }    
        }

        stringstream ss;
        ss << idx;
        putText(img_show, ss.str(), Point(0,60), 
                FONT_HERSHEY_PLAIN, 5, Scalar(0,255,0), 5);
        imshow("track", img_show);

        Mat img_show_seg;
        drawLabels(images_seg[idx], img_show_seg);
        imshow("seg", img_show_seg);
        switch( waitKey() )
        {            
            case 27: // ESC
            case 113: // q
                flag = false;
            break;
            case 97: // a
                idx = (idx-1 + n_images) % n_images;
            break;
            default:
                idx = (idx+1) % n_images;
            break;            
        }
    }

    return 0;
}

#if 0
    srv.request.regions.resize(2);
    srv.request.regions[0].x_offset = 126;
    srv.request.regions[0].y_offset = 311;
    srv.request.regions[0].width = 281 - 126;
    srv.request.regions[0].height = 413 - 311;

    srv.request.regions[1].x_offset = 457;
    srv.request.regions[1].y_offset = 236;
    srv.request.regions[1].width = 525 - 457;
    srv.request.regions[1].height = 356 - 236;

    vector<sensor_msgs::RegionOfInterest> &regions = srv.request.regions;
    vector<vector<Rect> > rects(n_images);
    for( size_t r=0; r<regions.size(); r++ )
    {
        rects[0].push_back(
            Rect( regions[r].x_offset,
                  regions[r].y_offset,
                  regions[r].width   ,
                  regions[r].height    ) );
    }
     
    for( size_t r=0; r<regions.size(); r++ )
    {   
        rectangle(images[0], rects[0][r], Scalar(0,255,0));
    }
    imshow("0", images[0]);

    clt_image_tracker.call(srv);
    for( int i=0; i<n_images-1; i++ )
    {        
        stringstream ss;
        ss << i+1;
    
        for( size_t r=0; r<regions.size(); r++ )
        {               
            Mat H(3,3,CV_64F,srv.response.homographies.data() + i*regions.size()*9 + r*9);

            cout << H << endl;

            Rect &rect = rects[i][r];            
            vector<Point2f> pts_src(4);
            pts_src[0].x = rect.x;            pts_src[0].y = rect.y;
            pts_src[1].x = rect.x+rect.width; pts_src[1].y = rect.y;
            pts_src[2].x = rect.x+rect.width; pts_src[2].y = rect.y+rect.height;
            pts_src[3].x = rect.x;            pts_src[3].y = rect.y+rect.height;

            vector<Point2f> pts_dst;
            perspectiveTransform(pts_src, pts_dst, H);

            float min_x= INFINITY, min_y= INFINITY, 
                  max_x=-INFINITY, max_y=-INFINITY;
            for( size_t p=0; p<pts_dst.size(); p++ )
            {
                if( min_x > pts_dst[p].x ) min_x = pts_dst[p].x;
                if( min_y > pts_dst[p].y ) min_y = pts_dst[p].y;
                if( max_x < pts_dst[p].x ) max_x = pts_dst[p].x;
                if( max_y < pts_dst[p].y ) max_y = pts_dst[p].y;
            }

            Rect rect_next(min_x, min_y, max_x-min_x, max_y-min_y);
            rects[i+1].push_back(rect_next);

            rectangle(images[i+1], rects[i+1][r], Scalar(0,255,0));            
        }

        imshow(ss.str(), images[i+1]);
    }
    waitKey();
#endif

#if 0
    Mat img_show(images[i].rows,images[i].cols+images[i+1].cols,CV_8UC3);
    images[i].copyTo(img_show(cv::Rect(0,0,images[i].cols,images[i].rows)));
    images[i+1].copyTo(img_show(cv::Rect(images[i].cols,0,
                                images[i+1].cols,images[i+1].rows)));

        line(img_show, pts_src[0], pts_src[1], Scalar(0,255,0));
        line(img_show, pts_src[1], pts_src[2], Scalar(0,255,0));
        line(img_show, pts_src[2], pts_src[3], Scalar(0,255,0));
        line(img_show, pts_src[3], pts_src[0], Scalar(0,255,0));

        vector<Point2f> pt2(4);
        pt2[0] = pts_dst[0]; pt2[0].x += images[0].cols;
        pt2[1] = pts_dst[1]; pt2[1].x += images[0].cols;
        pt2[2] = pts_dst[2]; pt2[2].x += images[0].cols;
        pt2[3] = pts_dst[3]; pt2[3].x += images[0].cols;

        line(img_show, pt2[0], pt2[1], Scalar(0,255,0));
        line(img_show, pt2[1], pt2[2], Scalar(0,255,0));
        line(img_show, pt2[2], pt2[3], Scalar(0,255,0));
        line(img_show, pt2[3], pt2[0], Scalar(0,255,0));
#endif