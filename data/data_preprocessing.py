"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).

Derived from https://github.com/chenxijun1029/DeepFM_with_PyTorch
"""
import os
import random
import collections

# There are 13 integer features and 26 categorical features
continuous_features = range(1, 14)
categorical_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continuous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def gen(self, idx, val):
        if val == '':
            return 0.0
        value = float(val)
        # if int(val) > continuous_clip[idx]:
        #     value = float(continuous_clip[idx])
        return value


def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    continuous_dict = ContinuousFeatureGenerator(len(continuous_features))

    categorical_dict = CategoryDictGenerator(len(categorical_features))
    categorical_dict.build(
        os.path.join(datadir, 'train.txt'), categorical_features, cutoff=200)

    dict_sizes = categorical_dict.dicts_sizes()

    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [1] * len(continuous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training
    with open(os.path.join(outdir, 'total_source.txt'), 'w') as out_source:
        with open(os.path.join(outdir, 'total_target_data.txt'), 'w') as out_target:
            with open(os.path.join(datadir, 'train.txt'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    continuous_vals = []
                    for i in range(0, len(continuous_features)):
                        val = continuous_dict.gen(i, features[continuous_features[i]])
                        continuous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    categorical_vals = []
                    for i in range(0, len(categorical_features)):
                        val = categorical_dict.gen(i, features[categorical_features[i]])
                        categorical_vals.append(str(val))

                    continuous_vals = ','.join(continuous_vals)
                    categorical_vals = ','.join(categorical_vals)
                    label = features[0]

                    if features[30] == "2005abd1":
                        out_target.write(','.join([continuous_vals, categorical_vals, label]) + '\n')
                    else:
                        out_source.write(','.join([continuous_vals, categorical_vals, label]) + '\n')


if __name__ == "__main__":
    preprocess('../data/dac', '../data')
