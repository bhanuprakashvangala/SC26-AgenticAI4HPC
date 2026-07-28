#!/bin/bash
echo "== ping 8.8.8.8 =="; ping -c1 -W2 8.8.8.8 2>&1 | tail -2
echo "== curl archive.ubuntu.com =="; curl -sI --max-time 6 http://archive.ubuntu.com/ubuntu/ 2>&1 | head -2
echo "== apt update tail =="; apt-get update 2>&1 | tail -5
