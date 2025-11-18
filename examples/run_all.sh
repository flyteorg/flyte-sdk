#!/bin/bash

for i in `ls *.py`
do
python $i
if [ "$?" -ne "0" ];
then
  break;
fi
done
