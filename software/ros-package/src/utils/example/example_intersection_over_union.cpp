#include <iostream>
#include <boost/program_options.hpp>

#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>

#include "utils/utils.hpp"

using namespace std;
using namespace pcl;
using namespace utils;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{   
    vector<string> fps_compare(2);
        
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("input1,i",   po::value<string>(&fps_compare[0]),
                         "first polymesh path to compare")
        ("input2,j",   po::value<string>(&fps_compare[1]),
                         "second polymesh path to compare")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if( vm.count("help")        ||        
        fps_compare.size()!=2      ) 
    {        
        for( size_t i=0; i<fps_compare.size(); i++ )
        {
            cout << fps_compare[i] << endl;
        }

        cout << desc << "\n";
        return 0;
    }

    PolygonMesh polymeshs[2];
    for( size_t i=0; i<2; i++ )
    {
        if(fps_compare[i].substr(fps_compare[i].length()-3) == "obj" &&
           pcl::io::loadPolygonFileOBJ(fps_compare[i], polymeshs[i]) == -1 )
        {
            cout << "[Error] Couldn't read obj file: " << fps_compare[i];
            return (-1);
        }

        if(fps_compare[i].substr(fps_compare[i].length()-3) == "ply" &&
           pcl::io::loadPLYFile(fps_compare[i], polymeshs[i]) == -1 )
        {
            cout << "[Error] Couldn't read ply file: " << fps_compare[i];
            return (-1);
        }
    }
    
    float AoU = IntersectionOverUnion(polymeshs[0], polymeshs[1] );
    cout << "AoU: " << AoU << endl;

    return 0;
}