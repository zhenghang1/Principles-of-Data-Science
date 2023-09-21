import os
import sys

dir = sys.argv[1]
if not os.path.isdir(dir):
    exit(1)

files = os.listdir(dir)
files = sorted(files)

distance = sys.argv[2]

result = []
for file in files:
    if not distance in file:
        continue
    path = os.path.join(dir,file)
    with open(path,'r') as f:
        last_line = f.readlines()[-1]
        acc = last_line.strip().split(' ')[-1][:-1]
        result.append((file,acc))
    
for f,a in result:
    print(f'{f}:\t{a}')