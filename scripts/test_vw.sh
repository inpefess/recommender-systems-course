#!/bin/sh
TOTAL=10
split --number l/${TOTAL} \
      --numeric-suffixes \
      --suffix-length 1 \
      test.vw test.
i=0
FINAL=$(expr $TOTAL - 1)
spanning_tree
while [ ${i} -le $FINAL ]
do
    if [ ${i} -eq $FINAL ]
    then
        BACKGROUND=
    else
        BACKGROUND="&"
    fi
    eval vw --span_server localhost \
       --total $TOTAL \
       --node ${i} \
       --unique_id 1 \
       --cache_file cache.${i} \
       --kill_cache \
       ${SAVE_MODEL} \
       --data test.${i} \
       --initial_regressor model.vw \
       --testonly \
       --predictions pred.${i} \
       > log.${i} 2>&1 $BACKGROUND
    i=$(expr ${i} + 1)
done
killall spanning_tree
cat pred.* > pred
