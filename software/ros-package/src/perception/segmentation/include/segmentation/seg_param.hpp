#ifndef SEG_PARAM__H__
#define SEG_PARAM__H__

#include <sstream>
#include <boost/tokenizer.hpp>

///  Default values of parameters before parsing
class SegParam {

public:
    SegParam(){};

    virtual void parse_param(const std::string &param)
    {
        boost::char_separator<char> sep(",");
        boost::tokenizer<boost::char_separator<char> > tok(param, sep);
        for( boost::tokenizer<boost::char_separator<char> >::iterator 
             beg=tok.begin(); beg!=tok.end(); beg++)
        {
            while( beg->compare("")==0 ) beg++;
            boost::char_separator<char> sep2("=");
            boost::tokenizer<boost::char_separator<char> > tok2(*beg, sep2);

            boost::tokenizer<boost::char_separator<char> >::iterator beg2=tok2.begin();
            std::string name = *beg2++;
            std::string val = *beg2++;
            set_param(name,val);
        }
    }

    virtual void print(std::stringstream &ss)=0;
    virtual void set_param(const std::string &name, const std::string &val)=0;

};

typedef class SuperVoxelParam : public SegParam {
    
public:
    SuperVoxelParam() : 
        voxel_resolution {0.0075f},
        seed_resolution {0.02f},
        color_importance {0.0f},
        spatial_importance {1.0f},
        normal_importance {1.0f},
        use_single_cam_transform {false},
        use_supervoxel_refinement {false},
        SegParam()
    {

    }

    // Supervoxel Param
    float voxel_resolution;
    float seed_resolution;
    float color_importance;
    float spatial_importance;
    float normal_importance;
    bool use_single_cam_transform;
    bool use_supervoxel_refinement;

    virtual void print(std::stringstream &ss)
    {
        ss << std::endl
        << "[ Supervoxel Param ]" << std::endl
        << "---------------------" << std::endl
        << "voxel_resolution: " << voxel_resolution << std::endl
        << "seed_resolution: " << seed_resolution << std::endl
        << "color_importance: " << color_importance << std::endl
        << "spatial_importance: " << spatial_importance << std::endl
        << "normal_importance: " << normal_importance << std::endl
        << "use_single_cam_transform: " << use_single_cam_transform << std::endl
        << "use_supervoxel_refinement: " << use_supervoxel_refinement << std::endl
        << "---------------------" << std::endl
        ;
    }

    virtual void set_param(const std::string &name, const std::string &val_str)
    {
        // std::string types
        float val;
        std::stringstream ss(val_str);
        ss >> val;            
        
        if(name.compare("voxel_resolution")==0)voxel_resolution=val;
        else if(name.compare("seed_resolution")==0)seed_resolution=val;
        else if(name.compare("color_importance")==0)color_importance=val;
        else if(name.compare("spatial_importance")==0)spatial_importance=val;
        else if(name.compare("normal_importance")==0)normal_importance=val;
        else if(name.compare("use_single_cam_transform")==0)use_single_cam_transform=val;
        else if(name.compare("use_supervoxel_refinement")==0)use_supervoxel_refinement=val;        
    }

} SuperVoxelParam;

typedef class LCCPParam : public SegParam {

public:
    LCCPParam() : 
        voxel_resolution {0.0075f},
        seed_resolution {0.03f},
        concavity_tolerance_threshold {10},
        smoothness_threshold {0.1},
        min_segment_size {0},
        use_extended_convexity {false},
        use_sanity_criterion {false},
        SegParam()
    {

    }

    // LCCPSegmentation Param
    float voxel_resolution;
    float seed_resolution;
    float concavity_tolerance_threshold;
    float smoothness_threshold;
    uint32_t min_segment_size;
    bool use_extended_convexity;
    bool use_sanity_criterion;

    virtual void print(std::stringstream &ss)
    {
        ss << std::endl
        << "[ LCCP Param ]" << std::endl
        << "------------------" << std::endl
        << "voxel_resolution: " << voxel_resolution << std::endl
        << "seed_resolution: " << seed_resolution << std::endl
        << "concavity_tolerance_threshold: " << concavity_tolerance_threshold << std::endl
        << "smoothness_threshold: " << smoothness_threshold << std::endl
        << "min_segment_size: " << min_segment_size << std::endl
        << "use_extended_convexity: " << use_extended_convexity << std::endl
        << "use_sanity_criterion: " << use_sanity_criterion << std::endl
        << "------------------" << std::endl
        ;
    }

    virtual void set_param(const std::string &name, const std::string &val_str)
    {
        // std::string types
        float val;
        std::stringstream ss(val_str);
        ss >> val;            
        
        if(name.compare("voxel_resolution")==0)voxel_resolution=val;
        else if(name.compare("seed_resolution")==0)seed_resolution=val;
        else if(name.compare("concavity_tolerance_threshold")==0)concavity_tolerance_threshold=val;
        else if(name.compare("smoothness_threshold")==0)smoothness_threshold=val;
        else if(name.compare("min_segment_size")==0)min_segment_size=val;
        else if(name.compare("use_extended_convexity")==0)use_extended_convexity=val;
        else if(name.compare("use_sanity_criterion")==0)use_sanity_criterion=val;        
    }

} LCCPParam;

typedef class LCCP2DSegParam : public SegParam {

public:
    LCCP2DSegParam() : 
        use_supervoxel {false},
        compute_face {true},
        compute_bbox {false},
        name_2dseg {"quickshift"},
        save_snapshot {false},
        remove_background {true},
        downsampling {true},
        num_of_objects {-1},
        SegParam()
    {
        strfmt_save = strfmt_save + getenv("HOME") + "/%06d.%s.%s";         
    }

    bool use_supervoxel;
    bool compute_face;
    bool compute_bbox;
    bool save_snapshot;
    bool remove_background;
    bool downsampling;
    int num_of_objects;
    std::vector<float> workspace;
    std::string name_2dseg;
    std::string strfmt_save;

    virtual void print(std::stringstream &ss)
    {
        ss << std::endl
        << "[ LCCP 2DSeg Param ]" << std::endl
        << "--------------------" << std::endl
        << "use_supervoxel: " << use_supervoxel << std::endl
        << "compute_face: " << compute_face << std::endl
        << "compute_bbox: " << compute_bbox << std::endl
        << "name_2dseg: " << name_2dseg << std::endl
        << "remove_background: " << remove_background << std::endl
        << "save_snapshot: " << save_snapshot << std::endl
        << "downsampling: " << downsampling << std::endl
        << "num_of_objects: " << num_of_objects << std::endl
        << "strfmt_save: " << strfmt_save << std::endl
        ;
        if( workspace.size() > 0 )
        {
            ss << "workspace: " 
               << "x=["<< workspace[0] << "," << workspace[1] << "] "
               << "y=["<< workspace[2] << "," << workspace[3] << "] "
               << "z=["<< workspace[4] << "," << workspace[5] << "] " 
               << std::endl;
        }
    }

    virtual void set_param(const std::string &name, const std::string &val_str)
    {
        std::cout << name <<  ":" << val_str << std::endl;

        std::stringstream ss(val_str);
        if(name.compare("use_supervoxel")==0)
        {
            bool val; ss >> val; use_supervoxel=val;
        }
        else if(name.compare("compute_face")==0)
        {            
            bool val; ss >> val; compute_face=val;
        }
        else if(name.compare("compute_bbox")==0)
        {            
            bool val; ss >> val; compute_bbox=val;
        }
        else if(name.compare("save_snapshot")==0)
        {            
            bool val; ss >> val; save_snapshot=val;
        }
        else if(name.compare("remove_background")==0)
        {
            bool val; ss >> val; remove_background=val;
        }
        else if(name.compare("downsampling")==0)
        {
            bool val; ss >> val; downsampling=val;
        }
        else if(name.compare("num_of_objects")==0)
        {
            int val; ss >> val; num_of_objects=val;
        }
        else if(name.compare("workspace")==0)
        {
            boost::char_separator<char> sep(" ");
            boost::tokenizer<boost::char_separator<char> > tok(val_str, sep);
            
            workspace.clear();
            for( boost::tokenizer<boost::char_separator<char> >::iterator 
                 it_tok=tok.begin(); it_tok!=tok.end(); it_tok++)
            {
                if( (*it_tok).compare("INFINITY")==0 ||
                    (*it_tok).compare("INF")==0      ||
                    (*it_tok).compare("infinity")==0 ||
                    (*it_tok).compare("inf")==0         )
                {
                    workspace.push_back(INFINITY);
                }
                else if( (*it_tok).compare("-INFINITY")==0 ||
                         (*it_tok).compare("-INF")==0      ||
                         (*it_tok).compare("-infinity")==0 ||
                         (*it_tok).compare("-inf")==0         )
                {
                    workspace.push_back(-INFINITY);
                }
                else
                {
                    std::stringstream ss2(*it_tok);                    
                    float val; ss2 >> val; workspace.push_back(val);
                }
            }
        }

        // std::string types
        if(name.compare("name_2dseg")==0)name_2dseg=val_str;
        else if(name.compare("strfmt_save")==0)strfmt_save=val_str;        
    }
} LCCP2DSegParam;

#endif
