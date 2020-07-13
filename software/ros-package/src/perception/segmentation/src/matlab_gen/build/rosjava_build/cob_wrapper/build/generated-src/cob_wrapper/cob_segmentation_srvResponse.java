package cob_wrapper;

public interface cob_segmentation_srvResponse extends org.ros.internal.message.Message {
  static final java.lang.String _TYPE = "cob_wrapper/cob_segmentation_srvResponse";
  static final java.lang.String _DEFINITION = "sensor_msgs/Image[] images_seg";
  static final boolean _IS_SERVICE = true;
  static final boolean _IS_ACTION = false;
  java.util.List<sensor_msgs.Image> getImagesSeg();
  void setImagesSeg(java.util.List<sensor_msgs.Image> value);
}
