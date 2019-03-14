# Google Cloud Storage Page
# https://console.cloud.google.com/storage/browser?_ga=2.228952860.-1660269020.1549912398&project=tpctpu&folder&organizationId

#Google Cloud Platform
#https://console.cloud.google.com/home/dashboard?project=tpctpu&cloudshell=true&_ga=2.60703405.-1660269020.1549912398
ctpu up
#ctpu up https://cloud.google.com/tpu/docs/ctpu-reference
export STORAGE_BUCKET=gs://tpctpu
gsutil cp -r ${STORAGE_BUCKET}/tpch1 ./

# gsutil cp -r  ./ ${STORAGE_BUCKET}

setsid tensorboard --logdir=${STORAGE_BUCKET}/res 
capture_tpu_profile --tpu=pedroholanda --logdir=${STORAGE_BUCKET}/res

pip install pandas