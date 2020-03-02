# encoding: utf-8

"""
Functions that help to select a GPU.
"""

import subprocess
import re
import os
import keras.backend as K


def parse_free_memory(output):
    """ Function to parse the output of nvidia-smi call.
    
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
    """ Used to get the amount of GPU's that reside in the machine
    
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
    """ Used to get a GPU index that satisfies the requested memory.

    :param claim_memory: A filter on the GPU's based on the amount of free memory available.
    Either a float in the range [0.0, 1.0], which represent a fraction of the total available GPU memory. 
    Or a float in the range (1.0, ...) that represents the desired amount of GPU memory.
    :return: index of gpu and percentage.
    """

    safety_factor = 0.9  # Used after a GPU has been selected.
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
        p_claim_memory = safety_factor * p_claim_mem
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
    """ Keras functionality to set and return a session with the given GPU settings.

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
    ## Example usage in Keras (tensorflow <2.0)
    index_gpu, p_gpu = get_free_gpu_id(claim_memory=0.5)
    sess = get_gpu_session(index_gpu, p_gpu)
    K.tensorflow_backend.set_session(sess)
       
    ## Example usage in Keras (tensorflow >= 2.0)
    # Though this approach might fail in the future because of current developments
    index_gpu, p_gpu = hnvidia.get_free_gpu_id(claim_memory=0.5)
    sess = hnvidia.get_gpu_session(index_gpu, p_gpu)
    tf.compat.v1.keras.backend.set_session(sess)

    ## Example usage Torch
    index_gpu, p_gpu = get_free_gpu_id(claim_memory=0.99)
    device = torch.device("cuda:{}".format(str(index_gpu)) if torch.cuda.is_available() else "cpu")
    model.to(device)
