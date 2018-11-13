#!/bin/bash
parallel ::: './a.out 0 100 111' './a.out 100 200 111' './a.out 200 300 111' './a.out 300 400 111' 
# parallel ::: './a.out 0 100 111' './a.out 100 200 111'
