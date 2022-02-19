#!/bin/sh
TOTAL=10
split --number l/${TOTAL} \
      --numeric-suffixes \
      --suffix-length 1 \
      test.vw test.
i=0
FINAL=$(expr $TOTAL - 1)
while [ ${i} -le $FINAL ]
do
    vw --span_server localhost \
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
       > log.${i} 2>&1 &
    i=$(expr ${i} + 1)
done
