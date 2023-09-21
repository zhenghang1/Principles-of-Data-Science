import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task1',exist_ok=True)
os.makedirs('output/task1/sigma_binary',exist_ok=True)

sigma = [1,5,15,10,20,25,30,35,40,45,50]
distance = ['euclidean','cityblock','chebyshev']

for s in sigma:
    for d in distance:
        output_path = f'output/task1/sigma_binary/{d}_sigma{s}.txt'
        if not os.path.exists(output_path):
            os.system(f'python main.py -t 1 --distance {d} -m binary -s {s} > {output_path}')