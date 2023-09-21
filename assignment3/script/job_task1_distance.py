import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task1',exist_ok=True)
os.makedirs('output/task1/distance_knn',exist_ok=True)


d_s = [('euclidean',0.5),('cityblock',0.5),('chebyshev',0.5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/distance_knn/{d}_continuous_nonorm.txt'
    if not os.path.exists(output_path):
        os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 0 --matrix_tpye continuous -c KNN > {output_path} &')

d_s = [('euclidean',40),('cityblock',45),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/distance_knn/{d}_continuous_norm.txt'
    if not os.path.exists(output_path):
        os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 1 --matrix_tpye continuous -c KNN > {output_path} &')


# binary
d_s = [('euclidean',10),('cityblock',45),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/distance_knn/{d}_binary_nonorm.txt'
    if not os.path.exists(output_path):
        os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 0 --matrix_tpye binary -c KNN > {output_path} &')


d_s = [('euclidean',15),('cityblock',50),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/distance_knn/{d}_binary_norm.txt'
    if not os.path.exists(output_path):
        os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 1 --matrix_tpye binary -c KNN > {output_path} &')
