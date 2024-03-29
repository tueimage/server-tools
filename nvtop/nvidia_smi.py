#! /usr/bin/env python
"""
Interface to nvidia-smi to gather all process pids for every NVIDIA-GPU on the 
system.

@author Koen Eppenhof
@date 2017/05/13
"""
from __future__ import print_function, division
import subprocess
import os

from pid import PIDInfo


class NvidiaSMI:
    """Process nvidia-smi output"""

    def __init__(self, test_mode=False):
        # If we are not in test mode.. then continue
        if test_mode:
            print('We have entered Test mode')
            with open('nvtop/nvidia_smi_L.txt', 'r') as f:
                self.gpu_names = f.read().split('\n')[:-1]
            with open('nvtop/nvidia_output_test.txt', 'r') as f:
                self.nvidia_smi = f.read()
        else:
            self.gpu_names = self._gpu_names()
            self.nvidia_smi = self._nvidia_smi()
            self.general_info = self._general_info()
            self.process_info = self._process_info()

    def _nvidia_smi(self):
        """Run nvidia-smi without extra options and returns output as str"""
        proc = subprocess.Popen(['nvidia-smi'],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE
                                )
        out, err = proc.communicate()
        if err:
            raise EnvironmentError('Failed to run nvidia-smi.')
        return out.decode()

    def _gpu_names(self):
        """Run nvidia-smi to list GPUs and returns GPU names as list of str"""
        proc = subprocess.Popen(['nvidia-smi', '-L'],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE
                                )
        out, err = proc.communicate()
        if err:
            raise EnvironmentError('Failed to run nvidia-smi.')
        return out.decode().split('\n')[:-1]

    def _general_info(self):
        """Parse the general info part of the nvidia-smi output and for each
        GPU extracts operating parameters (temperature, memory use, etc.),
        returned as a list with for each GPU a dictionary"""

        def parse(string):
            if 'N/' in string:
                return -1
            else:
                return string
        # Split the information of the GPU from the current processes
        header = self.nvidia_smi.split('Processes:')[0]
        lines = header.split('\n')

        # Lines 7:-3 actually contain the parameters
        starting_index = next(i for i, x in enumerate(lines) if x.startswith('|==='))
        gpu_info = lines[starting_index+1:-3]
        
        # In some output of nvidia-smi.. the amount of lines per GPU is increased
        # With this we account for the amount of lines that we need to read
        step_index = next(i for i, x in enumerate(gpu_info) if x.startswith('+---'))
        general_info = []
        for i in range(len(gpu_info) //(step_index+1)):
            name_line = gpu_info[(step_index+1) * i].split()
            info_line = [' '] + gpu_info[(step_index+1) * i + 1][1:].split()
            # The power cap is increased to 250W, leaving no white space between 'W' and |
            # This removes any '|' character in a potential string.
            info_line = [x.strip('|') for x in info_line]
            d = {}
            d['gpu_id'] = int(parse(name_line[1]))
            d['name'] = ' '.join(self.gpu_names[i].split()[:-2])
            d['persistence_m'] = parse(name_line[-7])
            d['bus_id'] = parse(name_line[-5])
            d['disp_a'] = parse(name_line[-4])
            d['uncorr_ecc'] = parse(name_line[-2])

            # Conditional statement to process Fan values 'ERR%'
            d['fan'] = int(info_line[1][:-1]) if info_line[1][:-1].isdigit() else -1
            d['temp'] = int(info_line[2][:-1])
            d['perf'] = info_line[3]
            d['pwr_usage'] = int(parse(info_line[4][:-1]))
            d['pwr_cap'] = int(parse(info_line[6][:-1]))
            d['mem_usage'] = int(parse(info_line[7][:-3]))
            d['mem_cap'] = int(parse(info_line[9][:-3]))
            d['gpu_util'] = int(parse(info_line[11][:-1]))
            d['comput_m'] = info_line[12]

            general_info.append(d)
        return general_info

    def _process_info(self):
        """Combine the information from nvidia-smi and the pid info for every
        pid, enabling printing the pid, username, gpu memory, ram, cpu etc."""
        process_lines = self.nvidia_smi.split('=====|')[-1].split('\n')[1:-2]
        process_list = []
        for line in process_lines:
            try:
                line_split = line.split()
                # In the new update nvidia-smi output we receive two more columns of info
                if len(line_split) == 9:
                    (_, gpu_id, gi, ci, pid, tp, name, gpu_mem, _) = line_split
                elif len(line_split) == 7:
                    (_, gpu_id, pid, tp, name, gpu_mem, _) = line_split
                else:
                    continue

                ps_line = PIDInfo(pid).info
                (user, pid, cpu, mem, vsz, rss, tty, stat, start, time) = \
                    ps_line.split()[:10]
                command = ' '.join(ps_line.split()[10:])
                process_list.append({
                    'gpu_id': gpu_id,
                    'pid': pid,
                    'type': tp,
                    'name': name,
                    'gpu_mem': gpu_mem[:-3],
                    'cpu': cpu,
                    'mem': mem,
                    'user': user,
                    'command': command
                })
            except ValueError:
                pass
        return process_list


if __name__ == '__main__':
    # print(NvidiaSMI().process_info())

    # # # Testing NvidiaSMI on the test text file
    A = NvidiaSMI(test_mode=True)
    A._general_info()
