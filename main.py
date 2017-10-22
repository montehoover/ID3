# ID3 algorithm for learning a decision tree for a target concept
import arff
import copy
from math import log2


def main():
    # Pre-processing
    with open('training_subsetD_small.arff') as f:
        training_data = arff.load(f)
    # examples = create_examples_list(training_data['data'], training_data['attributes'])
    # attributes = tupleslist_to_dict(training_data['attributes'])

    # Remove the target concept from the attributes dict
    del training_data['attributes'][-1]
    # attributes.pop('Class')



    # Learn the tree
    t = id3(training_data['data'], training_data['attributes'])
    if t: t.pretty_print()


    # for e in examples:
    # print()
    # print(examples[0].class_value)
    # print(predict(t, examples[0]))

    # print(information_gain(examples, attributes, 'wind'))
    # print(entropy_by_numbers(9,5))
    # print(entropy(examples))
    # print(split_information(examples, attributes, 'id'))
    # print(split_information(examples, attributes, 'wind'))


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

    # Recursive case:
    a = choose_best_attribute(examples, attributes)
    attributes_copy = copy.deepcopy(attributes)
    attributes_copy.pop(a, None)
    root.decision_attribute = a

    fill_in_unkown_values(examples, attributes, a)
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


def choose_best_attribute(examples, attributes):
    # Find the average Information Gain of all the attributes
    gains = []
    for i in range(len(attributes)):
        gains.append((information_gain(examples, attributes, i), a))
    avg_gain = sum([x[0] for x in gains])/float(len(gains))

    # Only consider those with above average gains and apply Split Information
    gain_ratios = []
    for g in gains:
        if g[0] >= avg_gain:
            # try:
            split_val = split_information(examples, attributes, g[1])
            if split_val != None:
                gain_ratio = g[0] / split_val
                gain_ratios.append((gain_ratio, g[1]))
            # except Exception as e:
            #     print("Mismatch between attribute and expected values. Continuing with remaining attributes.")
            #     print(e)

    if len(gain_ratios) == 0:
        raise Exception("None of attributes met criteria:", avg_gain, sorted(gains, reverse=True))

    # print(avg_gain, sorted(gains, reverse=True))
    # print(gain_ratios)

    # The attribute with best gain ratio:
    return max(gain_ratios)[1]


def information_gain(examples, attributes, attribute_index):
    subsets = split_by_attribute(examples, attributes, attribute_index)

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
        # raise Exception(
        #     "Split info was zero (no examples matched attribute). Attribute: {0}, Allowed values: {1}, Actual value from ex0: {2}".format(
        #         attribute, attributes[attribute], examples[0].attributes[attribute]))
    # return -sum


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

    num_positive_examples = 0
    num_negagive_examples = 0
    for e in examples:
        if e[-1] == 'True':
            num_positive_examples += 1
        else:
            num_negagive_examples += 1

    total = num_positive_examples + num_negagive_examples
    if target == 'True':
        return num_positive_examples/total
    elif target == 'False':
        return num_negagive_examples/total


# Divide the examples into subsets that correspond the each of the possible
# values for the decision_attribute.
# We return a list of subsets (one for each possible value), where each
# subset is represented by a two-item tuple. The first item is the value,
# and the second item is the list of examples that correspond to that
# value.
def split_by_attribute(examples, attributes, attribute_index):
    return [(value, [e for e in examples
                     if e[attribute_index] == value])
            for value in attributes[attribute_index][1]]


def fill_in_unkown_values(examples, attributes, attribute):
    for e in examples:
        if e.attributes[attribute] == None:
            e.attributes[attribute] = most_common_value(examples, attributes, attribute, e.class_value)


def most_common_value(examples, attributes, decision_attribute, class_value):
    d = {}
    for value in attributes[decision_attribute]:
        d[value] = 0
    for e in examples:
        if e.class_value == class_value and e.attributes[decision_attribute] != None:
            d[e.attributes[decision_attribute]] += 1

    return max(d.items(), key = lambda x: x[1])[0]


def predict(tree, example):
    if len(tree.children) == 0:
        print(tree.branch_value)
        return tree.label
    else:
        print(tree.branch_value, tree.decision_attribute)
        for child in tree.children:
            if example.attributes[tree.decision_attribute] == child.branch_value:
                print(child.branch_value)
                return predict(child, example)
        raise Exception("example value did match any valid attribute values",
                        example.attributes[tree.decision_attribute],
                        [x.branch_value for x in tree.children])


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