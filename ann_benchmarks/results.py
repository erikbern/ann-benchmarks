from __future__ import absolute_import

import h5py
import json
import os
import re


def get_algorithm_name(name, batch_mode):
    if batch_mode:
        return name + "-batch"
    return name


def is_batch(name):
    return "-batch" in name


def get_result_filename(dataset=None, count=None, definition=None,
                        query_arguments=None, batch_mode=False):
    d = ['results']
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(get_algorithm_name(definition.algorithm, batch_mode))
        data = definition.arguments + query_arguments
        d.append(re.sub(r'\W+', '_', json.dumps(data,
                                                sort_keys=True)).strip('_'))
    return os.path.join(*d)


def store_results(dataset, count, definition, query_arguments, attrs, results,
                  batch):
    fn = get_result_filename(
        dataset, count, definition, query_arguments, batch)
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    times = f.create_dataset('times', (len(results),), 'f')
    neighbors = f.create_dataset('neighbors', (len(results), count), 'i')
    distances = f.create_dataset('distances', (len(results), count), 'f')
    for i, (time, ds) in enumerate(results):
        times[i] = time
        neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
        distances[i] = [d for n, d in ds] + [float('inf')] * (count - len(ds))
    f.close()


def load_all_results(dataset=None, count=None, split_batched=False,
                     batch_mode=False):
    for root, _, files in os.walk(get_result_filename(dataset, count)):
        for fn in files:
            try:
                if split_batched and batch_mode != is_batch(root):
                    continue
                f = h5py.File(os.path.join(root, fn), 'r+')
                properties = dict(f.attrs)
                # TODO Fix this properly. Sometimes the hdf5 file returns bytes
                # This converts these bytes to strings before we work with them
                for k in properties.keys():
                    try:
                        properties[k] = properties[k].decode()
                    except:
                        pass
                yield properties, f
                f.close()
            except:
                pass


def get_unique_algorithms():
    algorithms = set()
    for properties, _ in load_all_results():
        algorithms.add(properties['algo'])
    return algorithms
