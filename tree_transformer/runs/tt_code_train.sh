export MAXTOKENS=1024
export INFER=y
export dis_port_str=--master_port=6102
export problem=nstack_merge_summ_cdds_65k
export MAX_UPDATE=61000
export UPDATE_FREQ=1
export att_dropout=0.2
export DROPOUT=0.3 &&

export CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$/\n/'`
echo "TT CODE TRAIN"
echo "CUDA VISIBLE DEVICES = $CUDA_VISIBLE_DEVICES"

bash run_code_nstack_nmt.sh dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier $CUDA_VISIBLE_DEVICES
