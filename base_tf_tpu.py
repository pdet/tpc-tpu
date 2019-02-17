import numpy as np
import timeit
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

column_1 = np.random.uniform(size=100000000)
column_2 = np.random.uniform(size=100000000)

def computation(c_1,c_2):
  d = tf.reduce_sum(tf.multiply(c_1, c_2))

def run(): 
  inputs = [tf.convert_to_tensor(column_1, np.float64),tf.convert_to_tensor(column_2, np.float64)]
  tpu_computation = tpu.rewrite(computation, inputs)
  tpu_grpc_url = TPUClusterResolver(
    tpu=[os.environ['TPU_NAME']]).get_master()
  with tf.Session(tpu_grpc_url) as sess:
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    res = sess.run(tpu_computation)
    sess.run(tpu.shutdown_system())
  return res

start_time = timeit.default_timer()
run()
print(timeit.default_timer() - start_time)