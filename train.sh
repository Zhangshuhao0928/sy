#!/bin/sh

mode='sy'
# mode='unet'
# mode='tcnn'

CUDA_VISIBLE_DEVICES=0 nohup python  sy_main.py "$mode" > log.txt 2>&1 &