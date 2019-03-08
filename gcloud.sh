export STORAGE_BUCKET=gs://tpctpu
gsutil cp -r ${STORAGE_BUCKET} ./

gsutil cp -r  ./ ${STORAGE_BUCKET}

setsid tensorboard --logdir=${STORAGE_BUCKET}/res 
capture_tpu_profile --tpu=pedroholanda --logdir=${STORAGE_BUCKET}/res