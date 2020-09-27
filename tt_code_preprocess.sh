# CODE SUMMARIZATION
export ROOT_DIR=`pwd`
export PROJDIR=tree_transformer
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_code_data
# export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-cdds
export valid_r=${RAW_DIR}/valid.tree-cdds
export test_r=${RAW_DIR}/test.tree-cdds
export OUT=${ROOT_DIR}/code_data_fairseq
rm -rf $OUT
python -m tree_transformer.preprocess_code_nstack2seq_merge \
--source-lang "cd" --target-lang "ds" \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 65536 --nwordstgt 65536 \
--no_remove_root \
--workers 8 \
--eval_workers 0 \
# --bpe_code ${BPE} \

# processed data saved in data_fairseq/nstack_merge_translate_ende_iwslt_32k