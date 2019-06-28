from experiment.execute import execute
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml
from utils.modelnames import models

import argparse
import pandas as pd
import timeit


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.tuning_result_path, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    R_test = load_numpy(path=args.path, name=args.test)

    R_train = R_train + R_valid

    topK = [5, 10, 15, 20, 50]

    frame = []
    for idx, row in df.iterrows():
        start = timeit.default_timer()
        row = row.to_dict()
        row['metric'] = ['R-Precision', 'NDCG', 'Precision', 'Recall', "MAP"]
        row['topK'] = topK
        result = execute(R_train, R_test, row, models[row['model']])
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Final")

    parser.add_argument('--name', dest='name', default="final_result.csv")
    parser.add_argument('--path', dest='path', default="data/yelp/")
    parser.add_argument('--tuning_result_path', dest='tuning_result_path', default='yelp')
    parser.add_argument('--test', dest='test', default='Rtest.npz')
    parser.add_argument('--train', dest='train', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid', default='Rvalid.npz')

    args = parser.parse_args()

    main(args)
