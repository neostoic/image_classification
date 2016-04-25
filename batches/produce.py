import hashlib
import pickle


class Produce:
    def __init__(self, id, jobq, doneq, seed, batch_fn, train_pkl, test_pkl, img_path, dtype, grayscale, pixels, model):
        while True:
            n = jobq.get()
            if n is None:
                break
            seed = hashlib.md5(seed + str(n)).hexdigest()
            seed = int(seed, 16) % 4294967295
            batch = batch_fn(seed, train_pkl=train_pkl, test_pkl=test_pkl, img_path=img_path,
                             dtype=dtype, grayscale=grayscale, pixels=pixels, model=model)
            with open('./run/shm/{}-{}'.format(id, n), 'w') as ofile:
                pickle.dump(batch, ofile, protocol=pickle.HIGHEST_PROTOCOL)
            doneq.put(n)
