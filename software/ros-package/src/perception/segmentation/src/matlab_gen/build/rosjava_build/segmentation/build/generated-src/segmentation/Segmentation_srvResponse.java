package segmentation;

public interface Segmentation_srvResponse extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "segmentation/Segmentation_srvResponse";
  static final java.lang.String _DEFINITION = "sensor_msgs/PointCloud2 segmented_cloud";
  static final boolean _IS_SERVICE = true;
  static final boolean _IS_ACTION = false;
  sensor_msgs.PointCloud2 getSegmentedCloud();
  void setSegmentedCloud(sensor_msgs.PointCloud2 value);
}
