model=$CAFFE_ROOT"/../data/models/alexnet/pretrained/places205CNN_iter_300000_upgraded.caffemodel"
network=$CAFFE_ROOT"/examples/domain_adaptation/network/alexnet/"
log_dir=$CAFFE_ROOT"/data/domain_adaptation_data/logs"
snapshot="snapshots_iter_1176.solverstate"
if [ "$1" = "finetune" ]
then
	solver=$network"pretrained/solver.prototxt"
	if [ "$2" = "default" ]
	then
		$CAFFE_ROOT/build/tools/caffe train --solver=$solver --weights=$model --log_dir=$log_dir -gpu 0
	elif [ "$2" = "trained" ]
	then
		$CAFFE_ROOT/build/tools/caffe train --solver=$solver --snapshot=$snapshot --log_dir=$log_dir -gpu 0
	else
		echo "wrong 2nd params "
	fi	
elif [ "$1" = "scratch" ]
then	
	solver=$network"scratch/solver.prototxt"
	$CAFFE_ROOT/build/tools/caffe train --solver=$solver --log_dir=$log_dir -gpu 0
else
	echo "wrong first params"
fi
