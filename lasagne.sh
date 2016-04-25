#!/bin/sh
#SBATCH -J convnet_keras             # Job name
#SBATCH -o convnet_keras.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -e convnet_keras.%j.err   # Name of stdout error file (%j expands to jobId)
#SBATCH -p normal           # Queue name
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 8                    # Total number of threads requestet
#SBATCH -t 99:00:00            # Run time (hh:mm:ss)


# run your script
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python /home/rcamachobarranco/code/convnet.py

deactivate
``````````````````````````End file````````````````````````````
