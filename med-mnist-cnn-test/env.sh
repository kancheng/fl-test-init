#!/bin/sh

if [ -d save ] && [ -d data ]; then
    echo "Directory exists."
else
    echo "Directory does not exists."
    mkdir save
    mkdir data
fi
