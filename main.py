# ID3 algorithm for learning a decision tree for a target concept
import collections

import arff
import copy
from math import log2
from scipy.stats import chi2

import pickle

import time


# Globals to be used for statistics:
POS_EXAMPLES = 0
NEG_EXAMPLES = 0
CONFIDENCE = 0


def main():
    start_time = time.time()
    # Pre-processing
    print("Reading file...")

    with open('training_subsetD.arff') as f:
        training_data = arff.load(f)

    print("Read file complete. Converting data to dicts...")

    examples = create_examples_list(training_data['data'], training_data['attributes'])
    attributes = tupleslist_to_dict(training_data['attributes'])
    # Remove the target concept from the attributes dict
    attributes.pop('Class')
    global POS_EXAMPLES, NEG_EXAMPLES
    POS_EXAMPLES, NEG_EXAMPLES = get_pos_and_neg_counts(examples)

    print("Converting data complete. Building the tree.")
    print(len(examples), "examples to start...")

    # Learn the tree
    t = id3(examples, attributes)
    # if t: t.pretty_print()
    train_end_time = time.time()
    print("Finished building tree in {} seconds".format(train_end_time - start_time))



    # run tests
    with open('testingD.arff') as f:
        test_data = arff.load(f)
    test_examples = create_examples_list(test_data['data'], test_data['attributes'])
    num_correct = 0
    num_incorrect = 0
    for e in test_examples:
        if predict(t, e) == e.class_value:
            num_correct += 1
        else:
            num_incorrect += 1
    percent_correct = num_correct / (num_correct + num_incorrect)
    print("Correct:", num_correct)
    print("Incorrect:", num_incorrect)
    print(percent_correct)


    test_end_time = time.time()
    print("Finished everything in {} seconds".format(test_end_time - start_time))


def id3(examples, attributes, branch_value=None):
    root = DecisionTreeNode(branch_value)

    # Validate input:
    if examples == None or len(examples) == 0 or attributes == None:
        raise Exception("Error: bad value passed. <examples>:",
                        examples, "<attributes>:", attributes)

    # Gather positive and negative probabilities for Base case 1 and
    # future use:
    most_common_value = None
    pos_probability = probability(examples, 'True')
    neg_probability = probability(examples, 'False')
    # (we bias to positive when split 50/50)
    if pos_probability >= neg_probability:
        most_common_value = 'True'
    else:
        most_common_value = 'False'

    # Base case 1:
    # If all examples are either positive or negative then assign that value to
    # the label
    if pos_probability == 1.0:
        root.label = 'True'  # Positive
        return root
    if neg_probability == 1.0:
        root.label = 'False'  # Negative
        return root

    # Base case 2:
    # If we are out of attributes to test on then assign the label the value of
    # the most common value in examples
    if len(attributes) == 0:
        root.label = most_common_value
        return root


    # Base case 3:
    # If no attributes pass the chi2 test for significance, then best_attribute will
    # be None. If so, stop splitting and create leaf node with most_common as label
    a = choose_best_attribute(examples, attributes)
    if a == None:
        root.label = most_common_value
        return root

    ####################
    # Recursive case:
    ####################

    # We want to continue growing the tree. Use returned best_attribute (a) to split.
    attributes_copy = copy.deepcopy(attributes)
    attributes_copy.pop(a, None)
    root.decision_attribute = a

    # Store most common value for that attribute among examples at this node
    val_true = get_most_common(examples, attributes, a, 'True')
    val_false = get_most_common(examples, attributes, a, 'False')
    root.most_common_value['True'] = val_true
    root.most_common_value['False'] = val_false
    fill_in_unknown_values(examples, a, val_true, val_false)

    # Split the data
    # Each subset is a tuple of decision_value and examples with that decision
    # value
    new_subsets = split_by_attribute(examples, attributes, a)
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
            print(len(subset[0]), "examples to go...")

            root.children[i] = id3(subset[1], attributes_copy, subset[0])

    return root


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
            split_val = split_information(examples, attributes, g[1])
            if split_val != None:
                gain_ratio = g[0] / split_val
                gain_ratios.append((gain_ratio, g[1]))

    if len(gain_ratios) == 0:
        raise Exception("None of attributes met criteria:", avg_gain, sorted(gains, reverse=True))

    # The attribute with best gain ratio:
    gain_ratios.sort(reverse=True)
    for val, attribute in gain_ratios:
        if is_statistically_significant(examples, attributes, attribute):
            return attribute

    return None


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
        return None
    else:
        return -sum


def entropy(examples):
    pos_probability = probability(examples, 'True')
    neg_probability = probability(examples, 'False')
    if pos_probability == 0 or neg_probability == 0:
        return 0
    else:
        return (-pos_probability*log2(pos_probability)
                - neg_probability*log2(neg_probability))


def probability(examples, target):
    if examples == None or len(examples) == 0:
        raise Exception("Error: Invalid argument for <examples>:", examples)
    pos, neg = get_pos_and_neg_counts(examples)
    total = pos + neg
    if target == 'True':
        return pos / total
    elif target == 'False':
        return neg / total


def get_pos_and_neg_counts(examples):
    num_positive_examples = 0
    num_negagive_examples = 0
    for e in examples:
        if e.class_value == 'True':
            num_positive_examples += 1
        else:
            num_negagive_examples += 1
    return num_positive_examples, num_negagive_examples


def is_statistically_significant(examples, attributes, attribute):
    subs = split_by_attribute(examples, attributes, attribute)
    test = independence_stat([x[1] for x in subs], POS_EXAMPLES, NEG_EXAMPLES)
    c = chi2.isf(1 - CONFIDENCE, len(subs) - 1)
    if test > c:
        return True
    else:
        return False

def independence_stat(subsets, p, n):
    sum = 0
    for subset in subsets:
        if len(subset) > 0:
            pi = 0
            ni = 0
            for e in subset:
                if e.class_value == 'True':
                    pi += 1
                else:
                    ni += 1
            ppi = pprimei(p, n, pi, ni)
            npi = nprimei(p, n, pi, ni)
            sum += (((pi - ppi) ** 2) / ppi) + (((ni - npi) ** 2) / npi)
    return sum

def pprimei(p, n, pi, ni):
    return p * (pi + ni) / (p + n)

def nprimei(p, n, pi, ni):
    return n * (pi + ni) / (p + n)


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


def fill_in_unknown_values(examples, attribute, class_true_val, class_false_val):
    for e in examples:
        if e.attributes[attribute] == None:
            if e.class_value == 'True':
                e.attributes[attribute] = class_true_val
            else:
                e.attributes[attribute] = class_false_val


def get_most_common(examples, attributes, decision_attribute, class_value):
    top_two = collections.Counter([e.attributes[decision_attribute] for e in [e for e in examples if e.class_value == class_value]]).most_common(2)

    if top_two[0][0] != None or len(top_two) == 1:
        return top_two[0][0]
    else:
        return top_two[1][0]


def predict(tree, example):
    if len(tree.children) == 0:
        # print(tree.branch_value)
        return tree.label
    else:
        # print(tree.branch_value, tree.decision_attribute)
        if example.attributes[tree.decision_attribute] == None:
            example.attributes[tree.decision_attribute] = tree.most_common_value[example.class_value]
        for child in tree.children:
            if example.attributes[tree.decision_attribute] == child.branch_value:
                # print(child.branch_value)
                return predict(child, example)

        # We get here if all the values were unknown during training, so we're just going random:
        return predict(tree.children[0], example)
        # raise Exception("example value did match any valid attribute values",
        #                 example.attributes[tree.decision_attribute],
        #                 [x.branch_value for x in tree.children])


def create_examples_list(example_tuples, attribute_tuples):
    examples = []
    for e in example_tuples:
        d = {}
        for j, a in enumerate(attribute_tuples):
            d[a[0]] = e[j]
        value = d.pop('Class')
        examples.append(Example(d, value))

    return examples


def tupleslist_to_dict(tl):
    d = {}
    for t in tl:
        d[t[0]] = t[1]
    return d


class DecisionTreeNode():
        def __init__(self, branch_value_from_parent):
            self.branch_value = branch_value_from_parent
            self.decision_attribute = None
            self.label = None
            self.children = []
            self.most_common_value = {}

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


class Example():
    def __init__(self, attribute_dict, class_value):
        self.attributes = attribute_dict
        self.class_value = class_value


if __name__ == '__main__':
    main()