# K-Nearest-Neighbors-Recommendation-on-Yelp
This repo implements and evaluates K-Nearest Neighbor model on Yelp Academic Dataset. 

# Example Commands

### Data Split
```
python getyelp.py --enable_implicit --ratio 0.5,0.2,0.3 --data_dir data/yelp/ --data_name yelp_academic_dataset_review.json
```

### Single Run
```
python main.py --path data/yelp/ --model KNN --k 90
```

### Hyper-parameter Tuning

Split data in experiment setting, and tune hyper parameters based on yaml files in `config` folder. 

```
python getyelp.py --enable_implicit --ratio 0.5,0.2,0.3 --data_dir data/yelp/ --data_name yelp_academic_dataset_review.json
python tune_parameters.py --parameters config/k_nearest_neighbor.yml --path data/yelp/ --save_path yelp/knn_tuning.csv
```

### Final Run
```
python final_performance.py --path data/yelp/ --tuning_result_path yelp --name final_result.csv
```

### Reference
* Scalable Collaborative Filtering with Jointly Derived Neighborhood
Interpolation Weights [[paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.218.109&rep=rep1&type=pdf)
