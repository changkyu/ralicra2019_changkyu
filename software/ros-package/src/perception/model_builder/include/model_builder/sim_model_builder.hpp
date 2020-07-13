#ifndef SIM_MODEL_BUILDER__H__
#define SIM_MODEL_BUILDER__H__

#include <vector>
#include "model_builder/vis_model_builder.hpp"

typedef class SimModelBuilder
{
public:
    typedef VisModelBuilder::VisConvexHull VisConvexHull;
    typedef shape_msgs::Plane Plane;

    SimModelBuilder();
    ~SimModelBuilder();

    void AddInputVisModel(VisConvexHull& chull);
    void AddInputVisModels(std::vector<VisConvexHull>& chulls);
    void AddStaticVisModel(Plane& planes);
    void AddStaticVisModels(std::vector<Plane>& planes);
    
    float TestVisModels();
    float TestVisModelsGui();
    float TestVisModelsTop2Bottom();
private:    
    std::vector<VisConvexHull> _chulls;
    std::vector<Plane>         _planes_bg;

} SimModelBuilder;

#endif