#!/usr/bin/env bash

LD_LIBRARY_PATH=../caffe_hed/install/lib:$LD_LIBRARY_PATH matlab -r ros_node -nodisplay
