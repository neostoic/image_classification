# Context manager to generate batches in the background via a process pool
# Usage:
#
# def batch(seed):
#    .... # generate minibatch
#    return minibatch
#
# with BatchGenCM(batch) as bg:
#    minibatch = next(bg)
#    .... # do something with minibatch

import os
import uuid
from multiprocessing import Process, Queue

import numpy as np

import produce


class BatchGen:
    def __init__(self, batch_fn, seed=None, num_workers=8, input_pkl='restaurant_photos_with_labels_train.pkl',
                 img_path='.', dtype='float64', grayscale=False, pixels=64, model='VGG_16', batch_size=32):
        self.batch_fn = batch_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        if seed is None:
            seed = np.random.randint(4294967295)
        self.seed = str(seed)
        self.id = uuid.uuid4()
        self.input_pkl = input_pkl
        self.img_path = img_path
        self.dtype = dtype
        self.grayscale = grayscale
        self.pixels = pixels
        self.model = model

    def __enter__(self):
        self.jobq = Queue(maxsize=self.num_workers)
        self.doneq = Queue()
        self.processes = []
        self.current_batch = 0
        self.finished_batches = []

        for i in range(self.num_workers):
            self.jobq.put(i)

            p = Process(target=produce.Produce, args=(
                self.id, self.jobq, self.doneq, self.seed, self.batch_fn, self.input_pkl, self.img_path,
                self.dtype, self.grayscale, self.pixels, self.model, self.batch_size))
            self.processes.append(p)
            p.start()

        return self

    def __iter__(self):
        return self

    def next(self):
        n = self.current_batch
        while n not in self.finished_batches:
            i = self.doneq.get()
            self.finished_batches.append(i)

        fn = './run/shm/{}-{}.npz'.format(self.id, n)
        batch = np.load(fn, mmap_mode='r')
        if os.name not in ['nt']:
            os.system('rm {}'.format(fn))
        else:
            os.system('DEL {}'.format(fn))
        self.jobq.put(n + self.num_workers)
        self.current_batch += 1
        return batch

    def __exit__(self, exc_type, exc_value, traceback):
        for _ in range(self.num_workers):
            self.jobq.put(None)
        for process in self.processes:
            process.join()
        while not self.doneq.empty():
            _ = next(self)
