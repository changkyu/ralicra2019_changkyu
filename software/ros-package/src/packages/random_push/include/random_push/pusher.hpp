#include <geometry_msgs/PoseStamped.h>
//#include "active_vision/PushCommand.h"
#include "TaskPlanner.hpp"

typedef class Pusher {
private:
    const char* gripperMode = "p";

    std::vector<double> init_pos = {0.5,0.0,0.7,-0.38268343, 0.92387953, 0.0, 0.0};
    geometry_msgs::PoseStamped target_pose;
    geometry_msgs::PoseStamped init_pose;

    TaskPlanner task_planner;

public:
    Pusher();
    Pusher(std::vector<double> &init_pos_in);
    bool Push(geometry_msgs::PoseStamped start_point,
              geometry_msgs::PoseStamped target_point);
    void GetInitPos(std::vector<double> &init_pos_out);
    void SetInitPos(std::vector<double> &init_pos_in);
    
    bool MoveHome();
    bool MoveInit();
    bool OpenFingers();
    bool CloseFingers();
    bool Move(geometry_msgs::PoseStamped pose);
    bool Move(double x, double y, double z);
    bool Steer(geometry_msgs::PoseStamped pose);
    bool Steer(double x, double y, double z);

    bool Push(double x_start, double y_start, double z_start,
              double x_end,   double y_end,   double z_end   );
    bool PrePush(double x_start, double y_start, double z_start,
                 double x_end,   double y_end,   double z_end   );
    
} Pusher;
