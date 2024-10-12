#!/bin/bash
# This script starts the application
echo "Starting the application..."
# Command to start your application
nohup python3 main.py > /dev/null 2>&1 &
