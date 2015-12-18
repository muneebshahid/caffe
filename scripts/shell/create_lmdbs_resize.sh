if [ "$1" = "train" ]
then
	lmdb_train=$PROJ_HOME"/data/lmdb/train/"
	lmdb_train1=$lmdb_train"train1"
	lmdb_train2=$lmdb_train"train2"
	rm -r $lmdb_train1
	rm -r $lmdb_train2
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/train/" $PROJ_HOME"/data/train/train1.txt" $lmdb_train1
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/train/" $PROJ_HOME"/data/train/train2.txt" $lmdb_train2
elif [ "$1" = "test" ]
then
	lmdb_test=$PROJ_HOME"/data/lmdb/test/"
	summer_lmdb=$lmdb_test"summer"
	winter_lmdb=$lmdb_test"winter"
	rm -r $summer_lmdb
	rm -r $winter_lmdb
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/test/" $PROJ_HOME"/data/test/summer.txt" $summer_lmdb
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/test/" $PROJ_HOME"/data/test/winter.txt" $winter_lmdb
else
	echo "Wrong Param"
fi
