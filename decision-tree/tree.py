import numpy as np

class Tree:
    def __init__(self, X_train, y_train, max_depth, min_samples):
        self.X_train = X_train
        self.max_depth = max_depth
        self.min_samples = min_samples

        root_node = Node()

        build_tree()
    
    def build_tree():
        if depth > max_depth or samples < min_samples:
            # Becomes leaf node
            print("hello")
        else:
            split_feature = choose_split_feature()


    # Returns column number for best feature to split on
    def choose_split_feature():
        return 1
    
class Node:
    def __init__(self, features, labels, left=None, right=None):
        self.features = features
        self.labels = labels
        self.left = left
        self.right = right
    


# Draw the tree
# Plot graphs, loss, etc. (separate class for this, can be reused)