# Google Cloud Storage Page
# https://console.cloud.google.com/storage/browser?_ga=2.228952860.-1660269020.1549912398&project=tpctpu&folder&organizationId

#Google Cloud Platform
#https://console.cloud.google.com/home/dashboard?project=tpctpu&cloudshell=true&_ga=2.60703405.-1660269020.1549912398

#Machine Options
#https://cloud.google.com/compute/docs/machine-types

#GPU Options
#https://cloud.google.com/compute/docs/gpus/
#https://cloud.google.com/compute/docs/gpus/add-gpus

#TPU Options
#https://cloud.google.com/tpu/docs/supported-versions#supported-tpu

#TPU ARTICLES
#https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu
#https://www.nextplatform.com/2018/05/10/tearing-apart-googles-tpu-3-0-ai-coprocessor/
ctpu up --tpu-size=v3-8 --machine-type=n1-standard-16
# pip install --install-option="--prefix=/export/scratch1/home/holanda/pythonlib" package_name
#PYTHONPATH="$PYTHONPATH:/export/scratch1/home/holanda/pythonlib/lib/python2.7"

# sudo apt-get install lshw
# sudo lshw -short

ctpu up
#ctpu up https://cloud.google.com/tpu/docs/ctpu-reference
export STORAGE_BUCKET=gs://tpctpu
gsutil cp -r ${STORAGE_BUCKET}/tpch-0.1 ./
gsutil cp -r ${STORAGE_BUCKET}/tpch1 ./
gsutil cp -r ${STORAGE_BUCKET}/tpch-10 ./
gsutil cp -r ${STORAGE_BUCKET}/tpch-100 ./

mv tpch1 tpch-1
gsutil cp -r ${STORAGE_BUCKET}/hyper ./
chmod 777 hyper/hyper.20182.18.1009.2120/hyperd
chmod 777 hyper/postgres-install/bin/psql
setsid hyper/hyper.20182.18.1009.2120/hyperd --database /home/pedroholanda/mydb --skip-license --no-ssl --no-password --log-dir /home/pedroholanda run
export PATH="/home/pedroholanda/hyper/postgres-install/bin:$PATH"
mkdir /home/pedroholanda/result/
sudo ln -s /home/pedroholanda/hyper/postgres-install/lib/libpq.so.5 /usr/lib/libpq.so.5
git clone https://github.com/pholanda/tpc-tpu.git
git clone https://github.com/pholanda/fakerep.git
psql -p 7483 -h localhost -U pedroholanda
/home/pedroholanda

# gsutil cp -r  ./ ${STORAGE_BUCKET}

setsid tensorboard --logdir=${STORAGE_BUCKET}/res 
capture_tpu_profile --tpu=pedroholanda --logdir=${STORAGE_BUCKET}/res --num_tracing_attempts=100

pip install pandas

exit

ctpu delete