from __future__ import absolute_import, division, print_function
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_datasets as tfds


tfds.disable_progress_bar()
LEARNING_RATE = 1e-4
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


CLUSTER_SPEC = {"worker": ["localhost:12345", "localhost:23456"]}


def start_server(job_name, task_index, tf_config):
    """ Create a server based on a cluster spec. """
    cluster = tf.train.ClusterSpec(CLUSTER_SPEC)
    server = tf.compat.v1.train.Server(
        cluster, config=tf_config, job_name=job_name, task_index=task_index)
    return server, cluster

def load_graph(filepath):
    with open(filepath) as f:
        txt = f.read()
        gdef = text_format.Parse(txt, tf.compat.v1.GraphDef())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(gdef, name="worker")

    return graph


job_name = "worker"
task_index = 1
# Set up tensorflow configuration.
tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# Start a server.
server, cluster = start_server(job_name, task_index, tf_config)


if job_name == "ps":
    server.join()
else:
    with tf.compat.v1.Session(target=server.target, graph=load_graph("./graph1.pbtxt")) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_op = sess.graph.get_operation_by_name("worker/GradientDescent/update_0_7/AssignAddVariableOp")
        loss = sess.graph.get_tensor_by_name("worker/Identity_2:0")

        _, train_loss = sess.run([train_op, loss])
        print(train_loss)

