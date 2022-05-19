import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.preprocessing import StandardScaler
import numpy as np

from performance_evaluation_with_ci import eval_result


class VdjMiner:
    def __init__(self, model_dir, data_dir, result_dir, target):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.target = target

    @staticmethod
    def _load_model(model_path):
        print('> load model ......')
        model = pickle.load(open(model_path, "rb"))
        return model

    def _load_data(self):
        print('> load data ......')
        data_path = os.path.join(self.data_dir, self.target + '.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def model_inference(self):
        model = self._load_model(os.path.join(self.model_dir, self.target + '.pkl'))
        datas = self._load_data()

        test_data = datas['test']
        features = [i for i in test_data.columns if i != self.target]
        test_feature, test_label = test_data.loc[:, features], test_data.loc[:, self.target]

        # scale test data
        train_feature = datas['train'].loc[:, features]
        scaler = StandardScaler()
        scaler.fit(train_feature)
        test_feature = scaler.transform(test_feature)
        print('> model inference ......')
        test_possibility = model.predict_proba(test_feature)
        return test_possibility, test_label

    def eval_result(self):
        test_possibility, test_label = self.model_inference()
        result_dir = self.result_dir
        target = self.target
        test_label = test_label.to_numpy()
        eval_result(test_possibility, test_label, result_dir, target, use_weight=True)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model_path", default="./models", help="model path")
    parser.add_argument("--data_dir", default="./data", help="test data path")
    parser.add_argument("--result_dir", default="results", help="result dir")
    parser.add_argument("--target", default="Cancer", help="target disease")
    return parser


def main(args):
    model_path = args.model_path
    data_dir = args.data_dir
    result_dir = args.result_dir
    target = args.target
    vdj_miner = VdjMiner(model_path, data_dir, result_dir, target)
    vdj_miner.eval_result()


if __name__ == '__main__':
    arg_parser = argparser()
    args, _ = arg_parser.parse_known_args()
    print(args)

    main(args)
