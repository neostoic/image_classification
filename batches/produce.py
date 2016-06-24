# Script that creates small batches of numpy by calling the batch function from load_batch.py. In this case batch_fn is
# passed as a string, thus it is not apparent that the batch function is being called
#
# This code was adapted from a gist from Eben Nolson, available at: https://gist.github.com/ebenolson/072712792c46aa192797

import hashlib

import numpy as np


class Produce:
    def __init__(self, id, jobq, doneq, seed, batch_fn, input_pkl, img_path, dtype, grayscale, pixels, model,
                 batch_size):
        while True:
            n = jobq.get()
            if n is None:
                break
            # Initialize seed for repeatability
            seed = hashlib.md5(str(seed) + str(n)).hexdigest()
            seed = int(seed, 16) % 4294967295
            # Call the batch function from load_batch.py
            x, y = batch_fn(seed, input_pkl=input_pkl, img_path=img_path, dtype=dtype, grayscale=grayscale,
                            pixels=pixels, model=model, batch_size=batch_size)

            # Pickle the current batch as compressed npz file
            np.savez_compressed('./run/shm/{}-{}'.format(id, n), x=x, y=y.astype('int32'))
            # Add to the queue
            doneq.put(n)
