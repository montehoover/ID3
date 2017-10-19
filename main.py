# ID3 algorithm for learning a decision tree for a target concept
from math import log2


class TennisExample():
    def __init__(self, outlook, temperature, humidity, wind, play_tennis):
        self.attributes = {
            'outlook': outlook,
            'temperature': temperature,
            'humidity': humidity,
            'wind': wind
        }
        self.target_concept = play_tennis


class AttributeMap():
    def __init__(self, values_map):
        self.possible_values = values_map

    def remove(self, attribute):
        self.possible_values.pop(attribute, None)


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


def entropy(pos_examples, neg_examples):
    pos_probability = pos_examples/(pos_examples+neg_examples)
    neg_probability = neg_examples/(pos_examples+neg_examples)
    if pos_probability == 0 or neg_probability == 0:
        return 0
    else:
        return (-pos_probability*log2(pos_probability)
                - neg_probability*log2(neg_probability))


def calculate_gain_ratio(attribute, examples):
    # TODO: fill in
    # entropy = 1.0
    # information_gain = None
    # distinct_examples = None
    # return information_gain / distinct_examples
    return 1


def choose_best_attribute(attributes, examples):
    attribute_gainratios = []
    for a in attributes.possible_values:
        gain_ratio = calculate_gain_ratio(a, examples)
        attribute_gainratios.append((gain_ratio, a))
    # The attribute with best gain ratio:
    print(attribute_gainratios)
    return max(attribute_gainratios)[1]


def split_by_attribute(decision_attribute, attributes, examples):
    # Divide the examples into subsets that correspond the each of the possible
    # values for the decision_attribute.
    # We return a list of subsets (one for each possible value), where each
    # subset is represented by a two-item tuple. The first item is the value,
    # and the second item is the list of examples that correspond to that
    # value.
    return [(value, [e for e in examples
                     if e.attributes[decision_attribute] == value])
            for value in attributes.possible_values[decision_attribute]]


def id3(examples, attributes, branch_value=None):
    root = DecisionTreeNode(branch_value)

    # Validate input:
    if examples == None or len(examples) == 0 or attributes == None:
        print("Error: bad value passed. <examples>:",
              examples, "<attributes>:", attributes)
        return None

    # Gather number of positive and negative examples for Base case 1 and
    # future use:
    num_positive_examples = 0
    num_negagive_examples = 0
    most_common_value = None
    for e in examples:
        if e.target_concept == True:
            num_positive_examples += 1
        else:
            num_negagive_examples += 1
    # (we bias to positive when split 50/50)
    if num_positive_examples >= num_positive_examples:
        most_common_value = True
    else:
        most_common_value = False

    # Base case 1:
    # If all examples are either positive or negative then assign that value to
    # the label
    if num_positive_examples == 0:
        root.label = False  # Negavite
        return root
    if num_negagive_examples == 0:
        root.label = True  # Positive
        return root

    # Base case 2:
    # If we are out of attributes to test on then assign the label the value of
    # the most common value in examples
    if len(attributes.possible_values) == 0:
        root.label = most_common_value
        return root

    # Recursive case:
    a = choose_best_attribute(attributes, examples)
    root.decision_attribute = a
    new_subsets = split_by_attribute(a, attributes, examples)
    print(new_subsets)
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
            attributes.remove(a)
            root.children[i] = id3(subset[1], attributes, subset[0])

    return root


def main():
    d1 = TennisExample('sunny', 'hot', 'high', 'weak', False)
    d2 = TennisExample('sunny', 'hot', 'high', 'strong', False)
    d3 = TennisExample('overcast', 'hot', 'high', 'weak', True)
    d4 = TennisExample('rain', 'mild', 'high', 'weak', True)
    d5 = TennisExample('rain', 'cool', 'normal', 'weak', True)
    d6 = TennisExample('rain', 'cool', 'normal', 'strong', False)
    d7 = TennisExample('overcast', 'cool', 'normal', 'strong', True)
    d8 = TennisExample('sunny', 'mild', 'high', 'weak', False)
    d9 = TennisExample('sunny', 'cool', 'normal', 'weak', True)
    d10 = TennisExample('rain', 'mild', 'normal', 'weak', True)
    d11 = TennisExample('sunny', 'mild', 'normal', 'strong', True)
    d12 = TennisExample('overcast', 'mild', 'high', 'strong', True)
    d13 = TennisExample('overcast', 'hot', 'normal', 'weak', True)
    d14 = TennisExample('rain', 'mild', 'high', 'strong', False)

    attributes = AttributeMap({
        'outlook': ['sunny', 'overcast', 'rain'],
        'temperature': ['hot', 'mild', 'cool'],
        'humidity': ['high', 'normal'],
        'wind': ['weak', 'strong']
    })

    examples = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]

    t = id3(examples, attributes)
    if t: t.pretty_print()

    print(entropy(6000,1))


if __name__ == '__main__':
    main()