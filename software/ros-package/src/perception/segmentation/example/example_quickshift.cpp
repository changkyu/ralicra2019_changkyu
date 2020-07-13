#include <boost/program_options.hpp>
#include <opencv2/highgui.hpp>

#include <segmentation/quickshift/quickshift_wrapper.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace cv;

static vector<Vec3b> colors;
void drawLabels(cv::Mat &input, cv::Mat &dst)
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
    string fp_image;
    float sigma;
    float tau;

    po::options_description desc("Example Quickshift Usage");
    desc.add_options()
        ("help", "help")
        ("image,i", po::value<string>(&fp_image),               "input image")
        ("sigma,s", po::value<float>(&sigma)->default_value(6), "sigma")
        ("tau,t", po::value<float>(&tau)->default_value(10),    "tau")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") || fp_image.compare("")==0 ) 
    {
        cout << desc << "\n";
        return 0;
    }

    Mat image = imread(fp_image);
    Mat image_seg;
    Segmentation_QuickShift(image, image_seg, sigma, tau);
    Mat image_show;
    drawLabels(image_seg, image_show);
    imshow("seg",image_show);
    waitKey();

    return 0;
}