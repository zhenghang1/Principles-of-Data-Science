import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task2',exist_ok=True)
os.makedirs('output/task2/binary',exist_ok=True)

d_s = [('euclidean',40),('cityblock',45),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task2/binary/{d}.txt'
    os.system(f'python main.py -t 2 --distance {d} --norm 0 --num_epoch 50 -m binary > {output_path} &')

    output_path = f'output/task2/binary/{d}_norm.txt'
    os.system(f'python main.py -t 2 --distance {d} --num_epoch 50 -m binary > {output_path} ')    
