import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task1',exist_ok=True)
os.makedirs('output/task1/knn',exist_ok=True)

d_s = [('euclidean',40),('cityblock',45),('chebyshev',5),('cos',1),('correlation',1)]

for d,s in d_s:
    output_path = f'output/task1/knn/{d}_knn.txt'
    os.system(f'python main.py -t 1 --distance {d} -s {s} -c KNN > {output_path} &')
