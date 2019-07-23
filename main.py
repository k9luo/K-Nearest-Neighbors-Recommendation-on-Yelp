from evaluation.metrics import evaluate
from models.predictor import predict
from utils.argcheck import check_float_positive, check_int_positive
from utils.io import load_numpy
from utils.modelnames import models
from utils.progress import WorkSplitter, inhour

import argparse
import numpy as np
import time


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.path))
    print("Train File Name: {}".format(args.train))
    if args.validation:
        print("Valid File Name: {}".format(args.valid))
    print("Algorithm: {}".format(args.model))
    print("Lambda Serendipity: {}".format(args.lambda_serendipity))
    print("Nearest Neighbor Number: {}".format(args.k))
    print("Evaluation Ranking Topk: {}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    R_train = load_numpy(path=args.path, name=args.train)
    print("Elapsed: {}".format(inhour(time.time() - start_time)))
    print("Train U-I Dimensions: {}".format(R_train.shape))

    progress.section("Train")
    model = models[args.model]()
    model.train(R_train)

    progress.section("Predict")
    prediction_score = model.predict(R_train,
                                     k=args.k,
                                     lambda_serendipity=args.lambda_serendipity)

    prediction = predict(prediction_score=prediction_score,
                         topK=args.topk,
                         matrix_Train=R_train)

    if args.validation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        R_valid = load_numpy(path=args.path, name=args.valid)
        result = evaluate(prediction, R_valid, metric_names, [args.topk])
        print("-")
        for metric in result.keys():
            print("{}:{}".format(metric, result[metric]))
        print("Elapsed: {}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Main")

    parser.add_argument('--disable_validation', dest='validation', action='store_false')
    parser.add_argument('--k', dest='k', type=check_int_positive, default=10)
    parser.add_argument('--model', dest='model', default="KNN")
    parser.add_argument('--path', dest='path', default="data/yelp/")
    parser.add_argument('--lambda_serendipity', dest='lambda_serendipity', type=check_float_positive, default=0)
    parser.add_argument('--topk', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('--train', dest='train', default='Rtrain.npz')
    parser.add_argument('--valid', dest='valid', default='Rvalid.npz')
    args = parser.parse_args()

    main(args)
