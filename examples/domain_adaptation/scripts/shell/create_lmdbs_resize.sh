convert_imageset="$CAFFE_ROOT/build/tools/convert_imageset"
data="$CAFFE_ROOT/../data/"
lmdb="$data/lmdb/"
images="$data/images/"

if [ $1 = "complete" ]
then
	complete_lmdb=$lmdb$1
	rm -r $complete_lmdb
	$convert_imageset -resize_height 256 -resize_width 256 $images $images$1".txt" $complete_lmdb
elif [ $1 = "source" ] || [ $1 = "target" ] || [ $1 = "train" ] || [ $1 = "test" ]
then
	lmdb1=$lmdb$1"1"
	lmdb2=$lmdb$1"2"
	rm -r $lmdb1
	rm -r $lmdb2
	$convert_imageset -resize_height 256 -resize_width 256 $images $images$1"1.txt" $lmdb1
	$convert_imageset -resize_height 256 -resize_width 256 $images $images$1"2.txt" $lmdb2
else
	echo "Wrong Param"
fi
