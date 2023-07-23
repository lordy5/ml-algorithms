import numpy as np

class Tree:
    def __init__(self, X_train, y_train, max_depth, min_samples):
        self.X_train = X_train
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.depth = 0
        self.samples = 0

        root = Node()
    
    def build_tree(self, node):
        if self.depth > self.max_depth or self.samples < self.min_samples:
            # Becomes leaf node
            print("hello")
        else:
            #split data and create child nodes
            left_node, right_node = node.split()
            # figure out the recursion for this
            self.build_tree(left_node)
            self.build_tree(right_node)



    # Returns column number for best feature to split on
    def choose_split_feature(self):
        return 1
    
class Node:
    def __init__(self, features, labels, left=None, right=None):
        self.features = features
        self.labels = labels
        self.left = left
        self.right = right

    # splits data, returns resulting left and right child nodes
    def split(self):
        split_row = self.get_split_row()
        # Split the features and labels
        features_left = self.features[:(split_row + 1), :]
        features_right = self.features[(split_row + 1):, :]
        labels_left = self.labels[:(split_row + 1), :]
        labels_right = self.labels[:(split_row + 1), :]

        left_child = Node(features_left, labels_left)
        right_child = Node(features_right, labels_right)
        
        return left_child, right_child



    # Returns row to split data at
    # Finds best feature and threshold to split at, and returns corresponding row
    # that splits at that threshold value
    def get_split_row(self):
        return 1, 1

        
    


# Draw the tree
# Plot graphs, loss, etc. (separate class for this, can be reused)