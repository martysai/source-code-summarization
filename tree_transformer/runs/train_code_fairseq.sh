#!/usr/bin/env bash

set -e

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# todo: specify gpus
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}

export ROOT_DIR=`pwd`
export PROJDIR=tree_transformer
export ROOT_DIR="${ROOT_DIR/\/tree_transformer\/runs/}"

export user_dir=${ROOT_DIR}/${PROJDIR}/src

export PROBLEM="${PROBLEM:-source-code-summarization}"

export RAW_DATA_DIR=${ROOT_DIR}/raw_code_data/${PROBLEM}
export DATA_DIR=${ROOT_DIR}/code_data_fairseq/${PROBLEM}
export TRAIN_DIR_PREFIX=${ROOT_DIR}/train_tree_transformer/${PROBLEM}


export ID="${ID:-1}"
export HPARAMS="${HPARAMS:-transformer_base}"

[ -z "$ARCH" ] && { echo "Must set export ARCH="; exit 1; } || echo "ARCH = ${ARCH}"


if [ ${HPARAMS} == "transformer_base" ]; then
    export TASK="${TASK:-translation}"

	export OPTIM=adam
	export ADAMBETAS='(0.85, 0.9)'
	export CLIPNORM=2.5
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
#	export
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-16000}"
	export LR="${LR:-0.00007}"
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.1}"
	export WDECAY="${WDECAY:-0.9}"
	export LB_SMOOTH=0.05
	export MAXTOKENS="${MAXTOKENS:-65536}" # 4096 -- 8gpus, 32768 -- 1gpu
	export UPDATE_FREQ="${UPDATE_FREQ:-64}" # 8 -- 8gpus, 1 -- gpu
#	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"

elif [ ${HPARAMS} == "transformer_base_stt2" ]; then

#	export MODEL=transformer
#	export HPARAMS=lm_gbw
#	export ARCH=${MODEL}_${HPARAMS}
	export ARCH="${ARCH:-fi_transformer_encoder_class_tiny}"
	export TASK="${TASK:-seq_classification}"
    export CRITERION="${CRITERION:-classification_cross_entropy}"

	export OPTIM="${OPTIM:-adam}"
	export ADAMBETAS='(0.85, 0.9)'
	export CLIPNORM=2.5
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-16000}"
	export LR="${LR:-0.00007}"
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.1}"
	export WDECAY=0.9
	export LB_SMOOTH=0.05
	export MAXTOKENS="${MAXTOKENS:-65536}" # 4096 -- 8gpus, 32768 -- 1gpu
	export UPDATE_FREQ="${UPDATE_FREQ:-64}"
	export MAX_UPDATE="${MAX_UPDATE:-2000}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-True}"
#	export LEFT_PAD_SRC=True
    export log_interval="${log_interval:-1000}"
    export max_sent_valid="--max-sentences-valid 1"
    export NCLASSES="${NCLASSES:-2}"
    export train_command="${train_command:-fairseq-train}"
else

    echo "undefined HPARAMS: ${HPARAMS}"
    exit 1
fi


export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"

export MAX_UPDATE="${MAX_UPDATE:-103000}"
export KEEP_LAS_CHECKPOINT="${KEEP_LAS_CHECKPOINT:-10}"

export DDP_BACKEND="${DDP_BACKEND:-c10d}"
export LRSRINK="${LRSRINK:-0.1}"
export MAX_LR="${MAX_LR:-0.001}"
export WORKERS="${WORKERS:-0}"
export INFER="${INFER:-n}"
export DISTRIBUTED="${DISTRIBUTED:-n}" # disable distributed
export CRITERION="${CRITERION:-label_smoothed_cross_entropy}"

export VALID_SET="${VALID_SET:-valid}"

export CRAWL_TEST="${CRAWL_TEST:-n}"
export extra_params="${extra_params:-}"

export RM_EXIST_DIR="${RM_EXIST_DIR:-n}"

export src="${src:-cd}"

export optim_text="${optim_text:---optimizer ${OPTIM} --clip-norm ${CLIPNORM} }"
export scheduler_text="${scheduler_text:---lr-scheduler ${LRSCHEDULE} --warmup-init-lr  ${WARMUP_INIT} --warmup-updates ${WARMUP} }"

export fp16="${fp16:-0}"
export rm_srceos="${rm_srceos:-0}"
export rm_lastpunct="${rm_lastpunct:-0}"
export nobar="${nobar:-1}"
export shareemb="${shareemb:-1}"
export shareemb_dec="${shareemb_dec:-1}"
export usetfboard="${usetfboard:-1}"

export dis_port_str="${dis_port_str:-}"
export nrank_str="${nrank_str:-}"

export max_sent_valid="${max_sent_valid:-}"

export att_dropout="${att_dropout:-0.03}"
# export weight_dropout="${weight_dropout:-0.03}"
#--max-sentences-valid 1

export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
export log_interval="${log_interval:-100}"

export TRAIN_DIR=${TRAIN_DIR_PREFIX}/${ARCH}-${HPARAMS}-b${MAXTOKENS}-gpu${NUM_GPU}-upfre${UPDATE_FREQ}-${fp16}fp16-id${ID}
export TFBOARD_DIR=${TRAIN_DIR}/tfboard

# todo: specify distributed and fp16
#[ ${fp16} -eq 0 ] && export fp16s="#" || export fp16s=
[ ${fp16} -eq 1 ] && export fp16s="--fp16 " || export fp16s=
[ ${rm_srceos} -eq 1 ] && export rm_srceos_s="--remove-eos-from-source " || export rm_srceos_s=
[ ${rm_lastpunct} -eq 1 ] && export rm_lastpunct_s="--remove-last-punct-source " || export rm_lastpunct_s=
[ ${nobar} -eq 1 ] && export nobarstr="--no-progress-bar" || export nobarstr=
[ ${NUM_GPU} -gt 1 ] && export distro= || export distro="#"

[ ${shareemb} -eq 1 ] && export shareemb_str="--share-all-embeddings " || export shareemb_str=
[ ${shareemb_dec} -eq 1 ] && export shareemb_dec_str="--share-decoder-input-output-embed " || export shareemb_dec_str=
[ ${usetfboard} -eq 1 ] && export tfboardstr="--tensorboard-logdir ${TFBOARD_DIR} " || export tfboardstr=

[ ${att_dropout} -eq 0 ] &&  export att_dropout_str= || export att_dropout_str="--attention-dropout ${att_dropout} "
[ ${weight_dropout} -eq 0 ] &&  export weight_dropout_str= || export weight_dropout_str="--weight-dropout ${weight_dropout} "


#--attention-dropout 0.1 \
#	--weight-dropout 0.1 \

rm -rf ${TRAIN_DIR}
echo "====================================================="
echo "START TRAINING: ${TRAIN_DIR}"
echo "PROJDIR: ${PROJDIR}"
echo "user_dir: ${user_dir}"
echo "ARCH: ${ARCH}"
echo "HPARAMS: ${HPARAMS}"
echo "DISTRO: ${distro}"
echo "INFER: ${INFER}"
echo "CRITERION: ${CRITERION}"
echo "fp16: ${fp16}"
echo "rm_srceos: ${rm_srceos}"
echo "rm_lastpunct_s: ${rm_lastpunct_s}"
echo "TFBOARD_DIR: ${TFBOARD_DIR}"
echo "====================================================="

if [ ${RM_EXIST_DIR} == "y" ]; then
    echo "Removing existing folder ${TRAIN_DIR}...."
    rm -rf ${TRAIN_DIR}
fi

mkdir -p ${TRAIN_DIR}

export out_log="${out_log:-n}"
export LOGFILE="${LOGFILE:-$TRAIN_DIR/train.log}"
export tee_begin=""
export tee_end=""

if [ $out_log == "y" ]; then
    echo "Printing logs to log file ${LOGFILE}"
    # export tee_begin=" -u "
    export tee_end=" | tee ${LOGFILE}"
    touch ${LOGFILE}
fi

#export last_params

export train_command="${train_command:-fairseq-train}"


if [ ${DISTRIBUTED} == "y" ]; then
    if [ ${NUM_GPU} -gt 1 ]; then
		echo "NUM GPU = $NUM_GPU > 1"
        export init_command="python ${tee_begin} -m torch.distributed.launch ${dis_port_str} --nproc_per_node ${NUM_GPU} $(which fairseq-train) ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} ${nrank_str}"
    else
		echo "NUM GPU = $NUM_GPU = 1"
        export init_command="$(which fairseq-train) ${tee_begin} ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} "
    fi
else
    export init_command="${train_command} ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} "
fi



echo "init_command = ${init_command}"


echo "Run model ${ARCH}, ${HPARAMS}"


if [ ${HPARAMS} == "transformer_base_stt2" ]; then
    export full_command="${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task ${TASK} \
	--valid-subset ${VALID_SET} \
	--source-lang ${src} \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
	--share-all-embeddings \
	${optim_text} \
	${scheduler_text} \
	--lr ${LR} \
	--min-lr ${MIN_LR} \
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion ${CRITERION} \
	--label-smoothing ${LB_SMOOTH} \
	--max-tokens ${MAXTOKENS} \
	--left-pad-source ${LEFT_PAD_SRC} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	--nclasses ${NCLASSES} \
	${max_sent_valid} \
	${extra_params} \
	${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s} | tee ${LOGFILE}"

else
    echo "run else commands"
	export full_command="${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task ${TASK} \
	--source-lang ${src} \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
	--optimizer ${OPTIM} \
	--clip-norm ${CLIPNORM} \
	--lr-scheduler ${LRSCHEDULE} \
	--warmup-init-lr  ${WARMUP_INIT} \
	--warmup-updates ${WARMUP} \
	--lr ${LR} \
	--min-lr ${MIN_LR} \
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion ${CRITERION} \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.85, 0.9)' \
	--max-tokens ${MAXTOKENS} \
	--left-pad-source ${LEFT_PAD_SRC} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	${att_dropout_str} \
	${weight_dropout_str} \
	${tfboardstr} \
	${shareemb_str} \
	${shareemb_dec_str} \
	${max_sent_valid} \
	${extra_params} \
	${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s} ${tee_end}"
fi

echo "full command: "
echo $full_command

eval $full_command

echo "=================="
echo "=================="
echo "finish training at ${TRAIN_DIR}"

if [ ${INFER} == "y" ]; then
    echo "Start inference ...."
    bash infer_code_model.sh
fi






