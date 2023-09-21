import os 

os.makedirs('output/',exist_ok=True)
os.makedirs('output/task1',exist_ok=True)
os.makedirs('output/task1/sigma_nonorm_svm',exist_ok=True)

sigma = [1,5,15,10,20,25,30,35,40,45,50]
distance = ['euclidean','cityblock','chebyshev']

for s in sigma:
    for d in distance:
        output_path = f'output/task1/sigma_nonorm_svm/{d}_sigma{s}_binary.txt'
        if not os.path.exists(output_path):
            os.system(f'python main.py -t 1 --distance {d} -m binary -s {s} --norm 0 > {output_path} &')

        output_path = f'output/task1/sigma_nonorm_svm/{d}_sigma{s}_continuous.txt'
        if not os.path.exists(output_path):
            os.system(f'python main.py -t 1 --distance {d} -s {s} --norm 0 > {output_path} ')