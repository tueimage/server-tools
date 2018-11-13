#! /usr/bin/env python
"""
Process 'ps' command output for given pid.

@author Koen Eppenhof
@date 2017/05/13
"""
from __future__ import print_function, division
import subprocess


class PIDInfo:
    """Process ps output for certain process ID (pid)"""
    def __init__(self, pid):
        ps_proc = subprocess.Popen(['ps', '-u', '-p', pid],
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE
                                   )
        out, err = ps_proc.communicate()
        self.info = out.decode().split('\n')[1]
