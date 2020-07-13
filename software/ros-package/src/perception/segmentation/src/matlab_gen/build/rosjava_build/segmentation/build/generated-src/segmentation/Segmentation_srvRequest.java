package segmentation;

public interface Segmentation_srvRequest extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "segmentation/Segmentation_srvRequest";
  static final java.lang.String _DEFINITION = "sensor_msgs/PointCloud2 input_cloud\n";
  static final boolean _IS_SERVICE = true;
  static final boolean _IS_ACTION = false;
  sensor_msgs.PointCloud2 getInputCloud();
  void setInputCloud(sensor_msgs.PointCloud2 value);
}
