from utils.argcheck import check_int_positive, ratio
from utils.io import get_yelp_df, save_array, save_numpy
from utils.progress import WorkSplitter
from utils.split import time_ordered_split

import argparse


def main(args):
    progress = WorkSplitter()

    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.data_dir))
    print("Implicit User Feedback: {}".format(args.implicit))

    progress.section("Load Raw Data")
    rating_matrix, timestamp_matrix = get_yelp_df(args.data_dir+args.data_name,
                                                  sampling=True,
                                                  top_user_num=args.top_user_num,
                                                  top_item_num=args.top_item_num)

    progress.section("Split CSR Matrices")
    rtrain, rvalid, rtest, nonzero_index, rtime = time_ordered_split(rating_matrix=rating_matrix,
                                                                     timestamp_matrix=timestamp_matrix,
                                                                     ratio=args.ratio,
                                                                     implicit=args.implicit)
    import ipdb; ipdb.set_trace()

    progress.section("Save NPZ")
    save_numpy(rtrain, args.data_dir, "Rtrain")
    save_numpy(rvalid, args.data_dir, "Rvalid")
    save_numpy(rtest, args.data_dir, "Rtest")
    save_numpy(rtime, args.data_dir, "Rtime")
    save_array(nonzero_index, args.data_dir, "Index")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="GetYelp")

    parser.add_argument('--data_dir', dest='data_dir', default="data/yelp/",
                        help='Directory path to the dataset. (default: %(default)s)')

    parser.add_argument('--data_name', dest='data_name', default='yelp_academic_dataset_review.json',
                        help='File name of the raw dataset. (default: %(default)s)')

    parser.add_argument('--enable_implicit', dest='implicit',
                        action='store_true',
                        help='Boolean flag indicating if user feedback is implicit.')

    parser.add_argument('--enable_sampling', dest='sampling',
                        action='store_true',
                        help='Boolean flag indicating if sampling is enabled.')

    parser.add_argument('--ratio', dest='ratio', default='0.5,0.2,0.3',
                        type=ratio,
                        help='Train, valid and test set split ratio. (default: %(default)s)')

    parser.add_argument('--top_item_num', dest='top_item_num', default=4000,
                        type=check_int_positive)

    parser.add_argument('--top_user_num', dest='top_user_num', default=6100,
                        type=check_int_positive)

    args = parser.parse_args()

    main(args)
