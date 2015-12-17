lmdb_train=$PROJ_HOME"/data/lmdb/train/"
lmdb_train1=$lmdb_train"train1"
lmdb_train2=$lmdb_train"train2"
rm -r $lmdb_train1
rm -r $lmdb_train2
$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/train/" $PROJ_HOME"/data/train/train1.txt" $lmdb_train1
$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/train/" $PROJ_HOME"/data/train/train2.txt" $lmdb_train2
