# ID3 algorithm for learning a decision tree for a target concept
import copy
from math import log2
from random import random

import pickle

import scipy


class TennisExample():
    def __init__(self, outlook, temperature, humidity, wind, id, region, play_tennis):
        self.attributes = {
            'outlook': outlook,
            'temperature': temperature,
            'humidity': humidity,
            'wind': wind,
            'id': id,
            'region': region
        }
        self.target_concept = play_tennis


class DecisionTreeNode():
    def __init__(self, branch_value_from_parent):
        self.branch_value = branch_value_from_parent
        self.decision_attribute = None
        self.label = None
        self.children = []

    def pretty_print(self):
        print('|')
        print(self.branch_value)
        print('|')
        if len(self.children) == 0:
            print('#', self.label, '#')
        else:
            print('*', self.decision_attribute, '*')
            for i, child in enumerate(self.children):
                print("Branch", i, ":")
                child.pretty_print()


def probability(examples, target):
    if examples == None or len(examples) == 0:
        raise Exception("Error: Invalid argument for <examples>:", examples)

    num_positive_examples = 0
    num_negagive_examples = 0
    for e in examples:
        if e.target_concept == True:
            num_positive_examples += 1
        else:
            num_negagive_examples += 1

    total = num_positive_examples + num_negagive_examples
    if target == True:
        return num_positive_examples/total
    elif target == False:
        return num_negagive_examples/total


def entropy(examples):
    pos_probability = probability(examples, True)
    neg_probability = probability(examples, False)
    if pos_probability == 0 or neg_probability == 0:
        return 0
    else:
        return (-pos_probability*log2(pos_probability)
                - neg_probability*log2(neg_probability))


def information_gain(examples, attributes, attribute):
    subsets = split_by_attribute(examples, attributes, attribute)

    new_entropy = 0
    # Each subset is represented by a tuple where we just care about the second
    # term (the actual list)
    for subset in subsets:
        if len(subset[1]) != 0:
            new_entropy += len(subset[1])/len(examples) * entropy(subset[1])

    return entropy(examples) - new_entropy


def split_information(examples, attributes, attribute):
    subsets = split_by_attribute(examples, attributes, attribute)
    sum = 0
    for subset in subsets:
        if len(subset[1]) != 0:
            sum += len(subset[1])/len(examples) * log2(len(subset[1])/len(examples))
    if sum == 0:
        raise Exception(
            "Split info was zero (no examples matched attribute). Attribute: {0}, Allowed values: {1}, Actual value from ex0: {2}".format(
                attribute, attributes[attribute], examples[0].attributes[attribute]))

    return -sum


def choose_best_attribute(examples, attributes):
    # Find the average Information Gain of all the attributes
    gains = []
    for a in attributes:
        gains.append((information_gain(examples, attributes, a), a))
    avg_gain = sum([x[0] for x in gains])/float(len(gains))

    # Only consider those with above average gains and apply Split Information
    gain_ratios = []
    for g in gains:
        if g[0] >= avg_gain:
            try:
                split_val = split_information(examples, attributes, g[1])
                gain_ratio = g[0] / split_val
                gain_ratios.append((gain_ratio, g[1]))
            except Exception as e:
                print("Mismatch between attribute and expected values. Continuing with remaining attributes.")
                print(e)

    if len(gain_ratios) == 0:
        raise Exception("None of attributes met criteria:", avg_gain, sorted(gains, reverse=True))

    # print(avg_gain, sorted(gains, reverse=True))
    # print(gain_ratios)

    # The attribute with best gain ratio:
    return max(gain_ratios)[1]


# Divide the examples into subsets that correspond the each of the possible
# values for the decision_attribute.
# We return a list of subsets (one for each possible value), where each
# subset is represented by a two-item tuple. The first item is the value,
# and the second item is the list of examples that correspond to that
# value.
def split_by_attribute(examples, attributes, decision_attribute):
    return [(value, [e for e in examples
                     if e.attributes[decision_attribute] == value])
            for value in attributes[decision_attribute]]


def id3(examples, attributes, branch_value=None):
    root = DecisionTreeNode(branch_value)

    # Validate input:
    if examples == None or len(examples) == 0 or attributes == None:
        raise Exception("Error: bad value passed. <examples>:",
                        examples, "<attributes>:", attributes)

    # Gather positive and negative probabilities for Base case 1 and
    # future use:
    most_common_value = None
    pos_probability = probability(examples, True)
    neg_probability = probability(examples, False)
    # (we bias to positive when split 50/50)
    if pos_probability >= neg_probability:
        most_common_value = True
    else:
        most_common_value = False

    # Base case 1:
    # If all examples are either positive or negative then assign that value to
    # the label
    if pos_probability == 1.0:
        root.label = True  # Positive
        return root
    if neg_probability == 1.0:
        root.label = False  # Negative
        return root

    # Base case 2:
    # If we are out of attributes to test on then assign the label the value of
    # the most common value in examples
    if len(attributes) == 0:
        root.label = most_common_value
        return root

    # Recursive case:
    a = choose_best_attribute(examples, attributes)
    attributes_copy = copy.deepcopy(attributes)
    attributes_copy.pop(a, None)
    root.decision_attribute = a

    new_subsets = split_by_attribute(examples, attributes, a)
    # Each subset is a tuple of decision_value and examples with that decision
    # value
    for i, subset in enumerate(new_subsets):
        # Create a placeholder for each new branch
        root.children.append(None)
        # If there weren't any examples for this branch (no info to test
        # against) then just give the most_common_value as a default, else
        # build a new tree based on this subset of examples
        if len(subset[1]) == 0:
            root.children[i] = DecisionTreeNode(subset[0])
            root.children[i].label = most_common_value
        else:
            root.children[i] = id3(subset[1], attributes_copy, subset[0])

    return root


def predict(tree, example):
    if len(tree.children) == 0:
        return tree.label
    else:
        for child in tree.children:
            if example.attributes[tree.decision_attribute] == child.branch_value:
                return predict(child, example)


def pprimei(p, n, pi, ni):
    return p*(pi+ni)/(p+n)

def nprimei(p, n, pi, ni):
    return n*(pi+ni)/(p+n)

def teststat(subsets, p, n):
    sum = 0
    for subset in subsets:
        pi = 0
        ni = 0
        for e in subset:
            if e.target_concept == True:
                pi += 1
            else:
                ni += 1
        ppi = pprimei(p, n, pi, ni)
        npi = nprimei(p, n, pi, ni)
        sum += ((pi - ppi)**2)/ppi + ((ni - npi)**2)/npi
    return sum

def main():
    n = 1
    examples = []
    examples.append(TennisExample('sunny', 'hot', 'high', 'weak',int(random()*n), 1, False))
    examples.append(TennisExample('sunny', 'hot', 'high', 'strong',int(random()*n), 2, False))
    examples.append(TennisExample('overcast', 'hot', 'high', 'weak',int(random()*n), 2, True))
    examples.append(TennisExample('rain', 'mild', 'high', 'weak', int(random()*n),2, True))
    examples.append(TennisExample('rain', 'cool', 'normal', 'weak', int(random()*n),2, True))
    examples.append(TennisExample('rain', 'cool', 'normal', 'strong', int(random()*n),2, False))
    examples.append(TennisExample('overcast', 'cool', 'normal', 'strong',int(random()*n), 2, True))
    examples.append(TennisExample('sunny', 'mild', 'high', 'weak', int(random()*n),2, False))
    examples.append(TennisExample('sunny', 'cool', 'normal', 'weak', int(random()*n),2, True))
    examples.append(TennisExample('rain', 'mild', 'normal', 'weak', int(random()*n),2, True))
    examples.append(TennisExample('sunny', 'mild', 'normal', 'strong',int(random()*n), 2, True))
    examples.append(TennisExample('overcast', 'mild', 'high', 'strong', int(random()*n),2, True))
    examples.append(TennisExample('overcast', 'hot', 'normal', 'weak', int(random()*n),2, True))
    examples.append(TennisExample('rain', 'mild', 'high', 'strong', int(random()*n),2, False))
    attributes = {
        'outlook': ['sunny', 'overcast', 'rain'],
        'temperature': ['hot', 'mild', 'cool'],
        'humidity': ['high', 'normal'],
        'wind': ['weak', 'strong'],
        'id': list(range(n)),
        'region': [1,2]
    }
    from scipy.stats import chi2




    # t = id3(examples, attributes)
    # if t: t.pretty_print()

    # for e in examples:
    #     print(predict(t, e))
    #
    # print([int(random()*n) for x in range(n)])
    # print()
    #
    # with open("tennis.pickle", 'wb') as f:
    #     pickle.dump(t, f)
    #
    # with open('tennis.pickle', 'rb') as f:
    #     t2 = pickle.load(f)
    #
    # # print(t == t2)
    # for e in examples:
    #     print(predict(t2, e))

    # t2.pretty_print()

    # print(information_gain(examples, attributes, 'wind'))
    # print(entropy_by_numbers(9,5))
    # print(entropy(examples))
    # print(split_information(examples, attributes, 'id'))
    # print(split_information(examples, attributes, 'wind'))



if __name__ == '__main__':
    main()