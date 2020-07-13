#include <random_push/pusher.hpp>

#include <Eigen/Geometry> 

using namespace std;

Pusher::Pusher()
: task_planner(gripperMode)
{
    SetInitPos(init_pos);
}

Pusher::Pusher(std::vector<double> &init_pos_in) 
: task_planner(gripperMode)
{
    SetInitPos(init_pos_in);
}

void Pusher::GetInitPos(std::vector<double> &init_pos)
{
    init_pos.resize(7);
    init_pos.at(0) = init_pose.pose.position.x;
    init_pos.at(1) = init_pose.pose.position.y;
    init_pos.at(2) = init_pose.pose.position.z;
    init_pos.at(3) = init_pose.pose.orientation.x;
    init_pos.at(4) = init_pose.pose.orientation.y;
    init_pos.at(5) = init_pose.pose.orientation.z;
    init_pos.at(6) = init_pose.pose.orientation.w;
}

void Pusher::SetInitPos(std::vector<double> &init_pos)
{
    init_pose.pose.position.x = init_pos.at(0);
    init_pose.pose.position.y = init_pos.at(1);
    init_pose.pose.position.z = init_pos.at(2);
    init_pose.pose.orientation.x = init_pos.at(3);
    init_pose.pose.orientation.y = init_pos.at(4);
    init_pose.pose.orientation.z = init_pos.at(5);
    init_pose.pose.orientation.w = init_pos.at(6);
}

bool Pusher::MoveHome()
{
    return task_planner.executeTask("HOME");
}

bool Pusher::MoveInit()
{
    return task_planner.executeTask("MOVE",init_pose);
}

bool Pusher::CloseFingers()
{
    return task_planner.executeTask("CLOSE_FINGERS");
}

bool Pusher::OpenFingers()
{    
    return task_planner.executeTask("TWO_FINGERS");
}

bool Pusher::Move(geometry_msgs::PoseStamped pose)
{
    return task_planner.executeTask("MOVE",pose);
}

bool Pusher::Move(double x, double y, double z)
{
    geometry_msgs::PoseStamped p = task_planner.get_current_pose();
    p.pose.position.x = x;
    p.pose.position.y = y;
    p.pose.position.z = z;
    return task_planner.executeTask("MOVE",p);
}

bool Pusher::Steer(geometry_msgs::PoseStamped pose)
{
    return task_planner.executeTask("STEER",pose);
}

bool Pusher::Steer(double x, double y, double z)
{
    geometry_msgs::PoseStamped p = task_planner.get_current_pose();
    p.pose.position.x = x;
    p.pose.position.y = y;
    p.pose.position.z = z;
    return task_planner.executeTask("STEER",p);
}

bool Pusher::PrePush(double x_start, double y_start, double z_start,
                     double x_end,   double y_end,   double z_end   )
{
    Eigen::Vector3f vec;
    vec << x_end - x_start, y_end - y_start, 0;
    vec = vec.normalized();

    float roll = M_PI, pitch = 0, yaw = atan(vec(1)/vec(0)) + M_PI/4;
    Eigen::Quaternion<float> q;    
    q =   Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ());

    geometry_msgs::PoseStamped p = task_planner.get_current_pose();
    p.pose.position.x = x_start;
    p.pose.position.y = y_start;
    p.pose.position.z = z_start + 0.2;    
    p.pose.orientation.x = q.x();
    p.pose.orientation.y = q.y();
    p.pose.orientation.z = q.z();
    p.pose.orientation.w = q.w();

    if( Steer(p) == false ) return false;

    return true;
}

bool Pusher::Push(double x_start, double y_start, double z_start,
                  double x_end,   double y_end,   double z_end   )
{
    if( Steer(x_start, y_start, z_start) == false) return false;

    if( Steer(x_end, y_end, z_end) == false) return false;

    if( Steer( init_pose.pose.position.x, 
               init_pose.pose.position.y, 
               init_pose.pose.position.z) == false) return false;

    return true;
}