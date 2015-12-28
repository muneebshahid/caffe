data=$CAFFE_ROOT"/data/domain_adaptation_data/"
convert_imageset=$CAFFE_ROOT"/build/tools/convert_imageset"
lmdb=$data"lmdb/"
images=$data"images/"

if [ $1 = "mean" ]
then
	complete_lmdb=$lmdb"complete"
	rm -r $complete_lmdb
	$convert_imageset -resize_height 256 -resize_width 256 $images $images"complete.txt" $complete_lmdb
elif [ $1 = "source" || $1 = "target" || $1 = "train" || $1 = "test" ]
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
