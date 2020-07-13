package cob_wrapper;

public interface cob_segmentation_srvRequest extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "cob_wrapper/cob_segmentation_srvRequest";
  static final java.lang.String _DEFINITION = "sensor_msgs/Image image\n";
  static final boolean _IS_SERVICE = true;
  static final boolean _IS_ACTION = false;
  sensor_msgs.Image getImage();
  void setImage(sensor_msgs.Image value);
}
