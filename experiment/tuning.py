from evaluation.metrics import evaluate
from models.predictor import predict
from tqdm import tqdm
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from utils.progress import WorkSplitter

import inspect
import numpy as np
import pandas as pd


def hyper_parameter_tuning(train, validation, params, save_path):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    try:
        df = load_dataframe_csv(table_path, save_path)
    except:
        df = pd.DataFrame(columns=['model', 'k', 'topK'])

    num_user = train.shape[0]

    for algorithm in params['models']:

        for k in params['k']:

                    if ((df['model'] == algorithm) &
                        (df['k'] == k)).any():
                        continue

                    format = "model: {}, k: {}"
                    progress.section(format.format(algorithm, k))

                    progress.subsection("Training")
                    model = params['models'][algorithm]()
                    model.train(train)

                    progress.subsection("Prediction")
                    prediction_score = model.predict(train, k=k)

                    prediction = predict(prediction_score=prediction_score,
                                         topK=params['topK'][-1],
                                         matrix_Train=train)

                    progress.subsection("Evaluation")
                    result = evaluate(prediction, validation, params['metric'], params['topK'])

                    result_dict = {'model': algorithm, 'k': k}

                    for name in result.keys():
                        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                    df = df.append(result_dict, ignore_index=True)

                    save_dataframe_csv(df, table_path, save_path)
