#!/bin/sh
TOTAL=10
RANK=10
split --number l/${TOTAL} \
      --numeric-suffixes \
      --suffix-length 1 \
      train.vw train.
i=0
FINAL=$(expr $TOTAL - 1)
while [ ${i} -le $FINAL ]
do
    if [ ${i} -eq $FINAL ]
    then   
	SAVE_MODEL="--final_regressor model.vw"
    else
        SAVE_MODEL=
    fi	   
    vw --span_server localhost \
       --total $TOTAL \
       --node ${i} \
       --unique_id 0 \
       --cache_file cache.${i} \
       --kill_cache \
       ${SAVE_MODEL} \
       --data train.${i} \
       --lrq ui${RANK} uu${RANK} ii${RANK} \
       --loss_function logistic \
       --learning_rate 0.001 \
       --passes 10 \
       --early_terminate 10 \
       --holdout_period 5 \
       > log.${i} 2>&1 &
    i=$(expr ${i} + 1)
done
