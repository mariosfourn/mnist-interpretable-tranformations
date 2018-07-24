"""
            used for cluster job submission and parameter search over main2.py, final model analysis

                """

import itertools
import os

import sys
import logging


def cartesian_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c):
    #     set_to_path = {
    #         'train': 'data/wn18/snli_1.0_train.jsonl.gz'
    #     }

    path = '/home/mfournar/mnist-interpretable-tranformations/02_Autoencoder_rotation'
    #     params = '-m cbilstm -b 32 -d 0.8 -r 300 -o adam --lr 0.001 -c 100 -e 10 ' \
    #              '--restore saved/snli/cbilstm/2/cbilstm -C 5000'
    #     command = 'PYTHONPATH=. python3-gpu {}/main.py {} ' \
    #     '--file_name {} ' \ this is for command if I want tensorboard

    command = "LD_LIBRARY_PATH='/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:\
    /share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python\
    -3.6.5-shared/lib:/share/apps/cuda-9.0/lib64:${LD_LIBRARY_PATH}'"+ "/share\
    /apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so $(command -v /share/apps\
    /python-3.6.5-shared/bin/python3) {}/main_penalty_loss.py  \
              --Lambda {} \
              --prop {} \
              --init-rot-range {}  \
              --relative-rot-range {}  \
              --epochs {} " \
        .format(path,
                #                 params,
                #                 set_to_path[c['instances']],
                c['Lambda'],
                c['prop'],
                c['init_rot_range'],
                c['relative_rot_range'],
                c['epochs']
                )
    return command


def to_logfile(c, path):
    outfile = "%s/penalty_loss.%s.log" % (path, summary(c))
    return outfile

def filename(c):
    outfile="penalty_loss_%s" %(summary(c))
    return outfile


def main(_):
    hyperparameters_space = dict(
        Lambda=[1.0,2.0], #Lambda
        prop=[1.0,2.0], #prop
        init_rot_range=[0,90], #init_rot_range
        relative_rot_range=[180,180], #relative_rot_range
        epochs= [20] #epochs
    )

    configurations = cartesian_product(hyperparameters_space)

    path = '/home/mfournar/output/logs/240718'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/mfournar/output'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '--name {} {} >> {} 2>&1'.format(filename(cfg),to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines,reverse=True)
    nb_jobs = len(sorted_command_lines)

# use #$ -pe smp 1000 for 1000 cores
 # add this in for GPU's   # $ -P gpu
 #    # $ -l gpu=

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /home/mfournar/output/array.o.log
#$ -e /home/mfournar/output/array.e.log
#$ -l tmem=16G
#$ -l h_rt=2:00:00
#$ -ac allow=LMNOPQSTU
#$ -l gpu=1
#$ -P gpu

#$ -j y



export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /home/mfournar/mnist-interpretable-tranformations/02_Autoencoder_rotation

source /share/apps/examples/python/python-3.6.5.source
source /share/apps/examples/cuda/cuda-9.0.source

    """.format(nb_jobs)

    print(header)

    #repeat each job three times

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id,command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])