data=$CAFFE_ROOT"/data/domain_adaptation_data/"
convert_imageset=$CAFFE_ROOT"/build/tools/convert_imageset"
lmdb=$data"lmdb/"
images=$data"images/"
if [ "$1" = "source" ]
then
	source1=$lmdb"source1"
	source2=$lmdb"source2"
	rm -r $source1
	rm -r $source2
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"source1.txt" $source1
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"source2.txt" $source2
elif [ "$1" = "target" ]
then
	target1=$lmdb"target1"
	target2=$lmdb"target2"
	rm -r $target1
	rm -r $target2
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"target1.txt" $target1
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"target2.txt" $target2
elif [ "$1" = "test" ]
then
	test1=$lmdb"test1"
	test2=$lmdb"test2"
	rm -r $test1
	rm -r $test2
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"test1.txt" $test1
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"test2.txt" $test2
elif [ "$1" = "mean" ]
then
	complete_lmdb=$lmdb"complete"
	rm -r $complete_lmdb
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"complete.txt" $complete_lmdb
else
	echo "Wrong Param"
fi
