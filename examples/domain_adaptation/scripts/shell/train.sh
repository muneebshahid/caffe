model=""
snapshot=""
network=$CAFFE_ROOT"/examples/domain_adaptation/network/alexnet/"
log_dir=$CAFFE_ROOT"/data/domain_adaptation_data/logs"
if [ "$1" = "finetune" ]
then
	solver=$network"pretrained/solver.prototxt"
	if [ "$2" = "default" ]
	then
		model=$CAFFE_ROOT"/../data/models/alexnet/pretrained/places205CNN_iter_300000_upgraded.caffemodel"		
	elif [ "$2" = "trained" ]
	then
		snapshot="$3"
	else
		echo "wrong 2nd params "
		exit
	fi
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --weights=$model --snapshot=$snapshot --log_dir=$log_dir -gpu 0
elif [ "$1" = "scratch" ]
then	
	solver=$network"scratch/solver.prototxt"
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --log_dir=$log_dir -gpu 0
else
	echo "wrong first params"
fi
