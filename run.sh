#!/bin/bash
###
 # @Author: Frank Liu
 # @Date: 2020-09-28 20:42:21
 # @LastEditors: Frank Liu
 # @LastEditTime: 2020-11-04 10:49:17
 # @Description: Auto training on GPU when it's free!
 # @FilePath: /DFFP/auto_gpu.sh
### 

while true
do
  count=$(ps -ef | grep -c kk_test)
  if [ $count -lt 4 ]
    then
      # 改动项 查询第1块gpu的容量--2p  第2块--3p  第三块--4p  第四块--5p 
      stat0=$(gpustat | awk '{print $11}' | sed -n '2p')
      stat1=$(gpustat | awk '{print $11}' | sed -n '3p')
      stat2=$(gpustat | awk '{print $11}' | sed -n '4p')
      stat3=$(gpustat | awk '{print $11}' | sed -n '5p')
      echo "gpu0: $stat0 M, gpu1: $stat1 M, gpu2: $stat2 M, gpu3: $stat3 M"
      if [ "$stat0" -lt 1000 ]
        then
          echo 'running on cuda:0...'
          python train_mi.py --gpu 0 --model efficientnet_b4 --epochs 300 --sufix _rms 
          exit 0
      elif [ "$stat1" -lt 1000 ]
        then
          echo 'running on cuda:1...'
          python train_mi.py --gpu 1 --model efficientnet_b4 --epochs 300 --sufix _rms 
          exit 0
      elif [ "$stat2" -lt 1000 ]
        then
          echo 'running on cuda:1...'
          python train_mi.py --gpu 2 --model efficientnet_b4 --epochs 300 --sufix _rms 
          exit 0
      fi
  fi
  sleep 2
done
