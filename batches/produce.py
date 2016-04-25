import hashlib

import numpy as np


class Produce:
    def __init__(self, id, jobq, doneq, seed, batch_fn, input_pkl, img_path, dtype, grayscale, pixels, model,
                 batch_size):
        while True:
            n = jobq.get()
            if n is None:
                break
            seed = hashlib.md5(str(seed) + str(n)).hexdigest()
            seed = int(seed, 16) % 4294967295
            x, y = batch_fn(seed, input_pkl=input_pkl, img_path=img_path, dtype=dtype, grayscale=grayscale,
                            pixels=pixels, model=model, batch_size=batch_size)

            np.savez_compressed('./run/shm/{}-{}'.format(id, n), x=x, y=y.astype('int32'))
            doneq.put(n)
