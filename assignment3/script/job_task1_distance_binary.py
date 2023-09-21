import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task1',exist_ok=True)
os.makedirs('output/task1/distance',exist_ok=True)

d_s = [('euclidean',15),('cityblock',50),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/distance/{d}_binary_norm.txt'
    os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 1 --matrix_tpye binary > {output_path} &')

    output_path = f'output/task1/distance/{d}_binary_nonorm.txt'
    os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 0 --matrix_tpye binary > {output_path} &')