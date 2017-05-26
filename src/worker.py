#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from gail_experimental import A3C_gail
from a3c import A3C
from envs import create_env
import distutils.version
import os
import yaml
from helpers import mkdir

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run(args, server):
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
    if args.cfg is not (None or False):
        # load the configuration filew
        cfg_dir = args.cfg +'.yaml'
        if os.path.exists(cfg_dir):
            pass
        else:
            cfg_dir = '../cfg/cfg.yaml'
        with open(cfg_dir, 'r+') as handle:
            cfg = yaml.load(handle)  # Data is in the form of a demonstration manager defined in another class.
            # if the path already exists load the cfg file in there.
        if cfg['demonstrations']!='None':
            trainer = A3C_gail(env, args.task, args.visualise, cfg)
        else:
            trainer = A3C(env, args.task, args.visualise, record=args.record)
    else:
        trainer = A3C(env, args.task, args.visualise, record = args.record)

    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if
                             not (v.name.startswith("local") or v.name.startswith("rif"))]
        for v in variables_to_save:
            print v.name
        print [v.name for v in tf.global_variables()]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if
                             not (v.name.startswith("local") or v.name.startswith("rif"))]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())


    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 53348207

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")


    #Make the cfg history directory.
    mkdir(args.log_dir+'/cfg_hist')
    # # save the cfg file used durign this restart of the training. One can change the main cfg file from the training directory. Nohting will be lost.
    hist_dir = args.log_dir+'/cfg_hist/'
    count = len([name for name in os.listdir(hist_dir) if os.path.isfile(name)])
    with open(hist_dir + "cfg_"+str(count)+".yaml", 'w+') as handle:
         yaml.dump(cfg,handle)


    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        #sess.run(trainer.reward_iface.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        print "GLOBAL STEPS", global_step
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            rollout = trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)
    os.system("tmux kill-session")


def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--cfg', type=str, default=False, help='configuration path')
    parser.add_argument('--record', action='store_true', default=False, help='Recording of trajectories enabled. '
                                                                             'If a demonstration path exists no recording takes place')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    tf.app.run()
