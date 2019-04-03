# encoding: utf-8

"""
Here we define all the necessary functions to choose a GPU device while using Keras Backend (Tensorflow)
"""

import subprocess
import re
import os
import keras.backend as K


def parse_free_memory(output):
    """
     $ nvidia-smi -q -d MEMORY -i 1
    ==============NVSMI LOG==============
    Timestamp                           : Wed Jan 31 14:13:22 2018
    Driver Version                      : 384.111
    Attached GPUs                       : 2
    GPU 00000000:06:00.0
        FB Memory Usage
            Total                       : 11172 MiB
            Used                        : 10 MiB
            Free                        : 11162 MiB
        BAR1 Memory Usage
            Total                       : 256 MiB
            Used                        : 2 MiB
            Free                        : 254 MiB

    :param output: String from nvidia-smi -q -d output
    :return: Free memory (in MiB) and total memory (in MiB)
    """

    pat_free = re.compile(r"Free\s+:\s+(\d+)\s+MiB")
    pat_total = re.compile(r"Total\s+:\s+(\d+)\s+MiB")

    m_free = int(pat_free.search(output).group(1))
    m_total = int(pat_total.search(output).group(1))

    return m_free, m_total


def get_num_gpu():
    """
    We can also use from tensorflow.python.client
        import device_lib
        device_lib.device_list()

    However this will allocate some memory on the GPUs themselves to check for something. So we'll use nvidia-smi -L

    :return: Number of GPUs by parsing nvidia-smi
    """
    res = subprocess.check_output("nvidia-smi -L", shell=True).decode('utf-8').split('\n')
    res = [x for x in res if len(x) > 0]
    return len(res)


def get_free_gpu_id(claim_memory):
    """
    nvidia-smi -q -d MEMORY -i

    :param claim_memory: either a fraction (of the whole gpu mem) or a set amount of MiB that you wish to occupy.
    :return: index of gpu and percentage.
    """

    # Used when not viable GPU was found under current settings
    safety_factor = 0.9
    ngpus = get_num_gpu()
    avail = [0] * ngpus
    total_avail = [0] * ngpus
    nvidia_cmd = ["nvidia-smi", "-q", "-d", "MEMORY", "-i"]

    # Loop over all available GPUs
    for i in range(ngpus):
        cmd = nvidia_cmd + [str(i)]
        output = subprocess.check_output(cmd).decode("utf-8")
        # Extract memory usage value
        avail[i], total_avail[i] = parse_free_memory(output)

    # Calculate usage percentage
    percent_avail = [x / y for x, y in zip(avail, total_avail)]

    # Set the right array from which we want to filter, based on claim_memory
    if claim_memory < 1:
        avail_mem = percent_avail
    else:
        avail_mem = avail

    # Make only those GPUs visible that satisfy the memory constraint imposed by claim_memory
    avail_mem_filter = [x if x >= claim_memory else 0 for x in avail_mem]

    # If nothing satisfies the condition, propose an alternative
    if all([x == 0 for x in avail_mem_filter]):
        print('No viable GPU was found. Choosing alternative option.')
        p_claim_memory = max(percent_avail)
        gpu_ind = percent_avail.index(p_claim_memory)
        p_claim_memory = safety_factor * p_claim_memory
        # Possibility to offer a choice of a new GPU
        # gpu_ind = -1
        # avail_mem_value = max(percent_avail)
        # alt_gpu_ind = percent_avail.index(avail_mem_value)
        # total_avail_value = total_avail[alt_gpu_ind]
        # p_claim_memory = {'alt_index_gpu': alt_gpu_ind,
        #                   'alt_avail_mem_perc': avail_mem_value,
        #                   'alt_avail_mem_mib': avail_mem_value*total_avail_value}

    else:
        avail_mem_value = max(avail_mem_filter)
        gpu_ind = avail_mem_filter.index(avail_mem_value)
        total_avail_value = total_avail[gpu_ind]

        # Translate the requested memory (in MiB) into a fraction
        if claim_memory < 1:
            p_claim_memory = claim_memory
        else:
            p_claim_memory = claim_memory/total_avail_value

    return gpu_ind, p_claim_memory


def get_gpu_session(index_gpu, p_gpu, verbose=True):
    """
    Sets and returns the session with the given param.

    :param index_gpu: the GPU that you want
    :param p_gpu: the fraction of total mem that you need
    :return: keras session that can be used for your model
    """
    res = subprocess.check_output("nvidia-smi -i " + str(index_gpu), shell=True).decode('utf-8')

    if verbose:
        print('Free GPU index: ', index_gpu)
        print('Current status of chosen GPU: ')
        print(res)

    # Choose the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index_gpu)

    # set this TensorFlow session as the default session for Keras
    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = p_gpu  ## This is of the total right..? Jep
    config.gpu_options.allow_growth = False  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = K.tf.Session(config=config)

    return sess


if __name__ == "__main__":
    # Example of usage
    index_gpu, p_gpu = get_free_gpu_id(claim_memory=0.5)
    sess = get_gpu_session(index_gpu, p_gpu)
    K.tensorflow_backend.set_session(sess)
