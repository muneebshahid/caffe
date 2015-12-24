if [ "$1" = "train" ]
then
	lmdb_train=$PROJ_HOME"/data/lmdb/train/"
	lmdb_train1=$lmdb_train"train1"
	lmdb_train2=$lmdb_train"train2"
	rm -r $lmdb_train1
	rm -r $lmdb_train2
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/images/" $PROJ_HOME"/data/images/train1.txt" $lmdb_train1
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/images/" $PROJ_HOME"/data/images/train2.txt" $lmdb_train2
elif [ "$1" = "test" ]
then
	lmdb_test=$PROJ_HOME"/data/lmdb/test/"
	test1_lmdb=$lmdb_test"test1"
	test2_lmdb=$lmdb_test"test2"
	rm -r $test1_lmdb
	rm -r $test2_lmdb
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/images/" $PROJ_HOME"/data/images/test1.txt" $test1_lmdb
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/images/" $PROJ_HOME"/data/images/test2.txt" $test2_lmdb
elif [ "$1" = "mean" ]
then
	complete_lmdb=$PROJ_HOME"/data/lmdb/complete/"
	rm -r $complete_lmdb
	$CAFFE_ROOT"/build/tools/convert_imageset" -resize_height 256 -resize_width 256 $PROJ_HOME"/data/images/" $PROJ_HOME"/data/images/complete.txt" $complete_lmdb
else
	echo "Wrong Param"
fi
