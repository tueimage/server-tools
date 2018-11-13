#! /usr/bin/env python
"""
Pretty print functions for a 'gpu card', showing for a given GPU the general 
information (fan speed, temperature, memory usage, etc.) and the processes
that run on that GPU, including the user.

@author Koen Eppenhof
@date 2017/05/13
"""
from __future__ import print_function, division
import subprocess
import os
import pwd

from nvidia_smi import NvidiaSMI


class Format:
    """Color codes for 256-color terminal"""
    BOLD = '\033[1;29m'
    DIM = '\033[2;29m'
    UNDERLINED = '\033[4;29m'
    INVERSE_BLUE = '\033[7;34m'
    INVERSE_WHITE_BOLD = '\033[1;29m\033[7;39m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    END = '\033[0;0m'


class Bar:
    """Print a colored progress bar that is green if it is filled for less than
    the threshold and turns red once it is over the threshold."""
    def __init__(self, size, threshold=0.75):
        self.size = size

    def draw(self, ratio):
        output = Format.END
        output += Format.UNDERLINED + Format.BOLD
        n = int(ratio * self.size)
        if ratio <= .75:
            output += Format.GREEN
        if ratio >= .75:
            output += Format.RED
        if ratio > 1:
            output += (self.size - 1) * '-' + '!'
        else:
            output += n * '-'
        output += Format.DIM + Format.UNDERLINED
        output += (self.size - n) * ' '
        output += Format.END
        return output


class GPUCard:
    """Print the info for a GPU, with both a header containing the current
    parameters and the process info for that GPU"""
    def __init__(self, gpu_id, nvidia_smi_output):
        nv = nvidia_smi_output
        self.info = nv.general_info()
        self.proc = nv.process_info()
        self.gpu_id = gpu_id
        _, line_width = os.popen('stty size', 'r').read().split()
        self.line_width = int(line_width)

    def title_part(self):
        gpu_info = self.info[self.gpu_id]
        name_line = ' {name:}'.format(**gpu_info)
        name_line += (self.line_width - len(name_line)) * ' '
        name_line = name_line[:-len(gpu_info['perf']) - 1]
        name_line += gpu_info['perf'] + ' \n'
        string = Format.INVERSE_WHITE_BOLD + \
            name_line[:self.line_width] + Format.END
        return string

    def header_part(self, compact=False):
        gpu_info = self.info[self.gpu_id]
        string = ''
        bar_width = 22

        if not compact:
            # GPU util bar (percentage)
            if gpu_info['gpu_util'] >= 0:
                string += Format.DIM + '   Util: ' + \
                    Bar(bar_width // 2).draw(float(gpu_info['gpu_util']) / 100.) + \
                    Format.END + \
                    '{:>5.0%}'.format(int(gpu_info['gpu_util']) / 100.)
            else:
                string += Format.DIM + '   Util: ' + \
                    Bar(bar_width // 2).draw(0) + Format.END + '   N/A '

            # GPU temperature bar (percentage of 100C)
            if gpu_info['temp'] >= 0:
                string += Format.DIM + '   Temp: ' + \
                    Bar(bar_width // 2).draw(gpu_info['temp'] / 100.) + \
                    Format.END + \
                    '{:>4d}C'.format(gpu_info['temp']) + Format.END
            else:
                string += Format.DIM + '   Temp: ' + \
                    Bar(bar_width // 2).draw(0) + Format.END + '  N/A '

            # GPU fan bar (percentage)
            if gpu_info['fan'] >= 0:
                string += Format.DIM + '    Fan: ' + \
                    Bar(bar_width // 2).draw(gpu_info['fan'] / 100.) + \
                    Format.END + \
                    '{:>5d}%'.format(gpu_info['fan']) + Format.END
            else:
                string += Format.DIM + '    Fan: ' + \
                    Bar(bar_width // 2).draw(0) + Format.END + '  N/A '

            string += '\n'

            # GPU power usage bar (percentage)
            if gpu_info['pwr_usage'] >= 0 and gpu_info['pwr_cap'] >= 0:
                pwr_ratio = gpu_info['pwr_usage'] / gpu_info['pwr_cap']
                string += Format.DIM + '  Power: ' + Bar(36).draw(pwr_ratio) + \
                    Format.END + '{:5.0%}'.format(pwr_ratio) + \
                    '    ' + str(gpu_info['pwr_usage']) + ' / ' + \
                    str(gpu_info['pwr_cap']) + 'W' + \
                    Format.END
            else:
                string += Format.DIM + '  Power: ' + Bar(36).draw(0) + \
                    Format.END + ' / N/A ' + Format.END

            string += '\n'

        # GPU memory usage bar (percentage)
        if gpu_info['mem_usage'] >= 0 and gpu_info['mem_cap'] >= 0:
            mem_ratio = gpu_info['mem_usage'] / gpu_info['mem_cap']
            string += Format.DIM + '   VRAM: ' + Bar(36).draw(mem_ratio) + \
                Format.END + '{:5.0%}'.format(mem_ratio) + \
                '    ' + str(gpu_info['mem_usage']) + ' / ' + \
                str(gpu_info['mem_cap']) + 'MiB' + \
                Format.END
        else:
            string += Format.DIM + ' Memory: ' + Bar(36).draw(0) + \
                Format.END + '  N/A ' + Format.END

        return string.format(**gpu_info)

    def process_part(self, compact=False):
        proc_for_gpu = [line for line in self.proc
                        if line['gpu_id'] == str(self.gpu_id)]
        format_str = (' {user:9s} {pid:>6s}   {cpu:>4s}   {mem:>4s}'
                      '   {gpu_mem:>10s}   {command:100s}'
                      )
        format_str_user = (Format.BLUE + Format.BOLD + ' {user:9s}'
                           ' {pid:>6s}   {cpu:>4s}   {mem:>4s}'
                           '   {gpu_mem:>10s}   {command:100s}'
                      )
        header_line = format_str.format(user='User', pid='PID', cpu='%CPU',
                                        mem='%RAM', gpu_mem='VRAM (MiB)',
                                        command='Command')[:self.line_width]
        header_line += (self.line_width - len(header_line) - 1) * ' ' + '\n'

        string = ' ' + Format.END + Format.UNDERLINED + Format.BOLD + \
            header_line[1:] + Format.END

        if len(proc_for_gpu) == 0:
            string += Format.PURPLE + Format.BOLD + ' No processes' + Format.END
            if not compact: string += '\n'

        for line in proc_for_gpu:
            line = self.process_command_commentary(line)
            if line['user'] == self.get_username():
                string += Format.END + \
                    format_str_user.format(**line)[:self.line_width - 1] + \
                    Format.END + '\n'
            else:
                string += Format.END + \
                    format_str.format(**line)[:self.line_width - 1] + \
                    Format.END + '\n'
        return string

    def process_command_commentary(self, line):
        if '<!' in line['command']:
            sp = line['command'].split('<!')
            before = sp[0]
            after = '<!' + '<!'.join(sp[1:])
            line['command'] = before + '  ' + after
            if '<!!!' in line['command']:
                line['command'] = line['command'].replace('<!!!',
                    Format.END + Format.RED)
            if '<!!' in line['command']:
                line['command'] = line['command'].replace('<!!',
                    Format.END + Format.YELLOW)
            if '<!' in line['command']:
                line['command'] = line['command'].replace('<!',
                    Format.END + Format.GREEN)
            if '!!!>' in line['command']:
                line['command'] = line['command'].replace('!!!>',
                    Format.END)
            if '!!>' in line['command']:
                line['command'] = line['command'].replace('!!>',
                    Format.END)
            if '!>' in line['command']:
                line['command'] = line['command'].replace('!>',
                    Format.END)
        return line

    def show(self, compact=False):
        print(self.header_part(compact))
        print(self.process_part(compact))

    def get_username(self):
        return pwd.getpwuid(os.getuid())[0]

