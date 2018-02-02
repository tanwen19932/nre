#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : fileutil.py
# @Author: TW
# @Date  : 2018/2/2
# @Desc  :
import json, os
def list_dir(path):
  files= []
  for i in os.listdir(path):
    temp_dir = os.path.join(path, i)
    if os.path.isdir(temp_dir):
      temp = {"dirname": temp_dir, 'child_dirs': [], 'files': []}
      files.extend(list_dir(temp_dir, temp))
    else:
      files.append(temp_dir)
  return files

if __name__ == '__main__':
    print(list_dir("../data/model"))

