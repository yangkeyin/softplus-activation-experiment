import pickle

with open('experiment_results/results.pkl', 'rb') as f:
    results = pickle.load(f)
    print(results[100][32][0.5][42])
