import numpy as np

# Classification decision tree
class Tree:
    def __init__(self, X_train, y_train, max_depth, min_samples):
        self.X_train = X_train
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.root = Node(0)
    
    def build_tree(self, node):
        if self.root.depth > self.max_depth or self.root.samples < self.min_samples:
            '''Criteria for splitting is no longer met, so create a leaf node in the tree.
            To do this, we find the majority class label in the node's data, and assign that
            label to the node.'''
            node.set_class()
        else:
            #split data and create child nodes
            left_node, right_node = node.split()
            # figure out the recursion for this
            node.build_tree(left_node)
            node.build_tree(right_node)



    # Returns column number for best feature to split on
    def choose_split_feature(self):
        return 1
    
class Node:
    def __init__(self, depth, features, labels, left=None, right=None):
        self.features = features
        self.labels = labels
        self.left = left
        self.right = right

        self.depth = depth
        self.samples = np.shape(features)[0]

    # splits data, returns resulting left and right child nodes
    def split(self):
        split_row = self.get_split_row()
        # Split the features and labels
        features_left = self.features[:(split_row + 1), :]
        features_right = self.features[(split_row + 1):, :]
        labels_left = self.labels[:(split_row + 1), :]
        labels_right = self.labels[:(split_row + 1), :]

        left_child = Node(self.depth + 1, features_left, labels_left)
        right_child = Node(self.depth + 1, features_right, labels_right)
        
        return left_child, right_child
    
    # If node is a leaf, find and set the class
    def set_class(self):
        # Find the label with the most non-zero entries (Assuming each label has a column,
        # where a 1 is true and 0 is false for that class)
        nonzero_counts = np.count_nonzero(self.labels, axis=0)
        self.leaf_label = np.argmax(nonzero_counts)
        self.is_leaf = True



    # Returns row to split data at
    # Finds best feature and threshold to split at, and returns corresponding row
    # that splits at that threshold value
    def get_split_row(self):
        return 1, 1

        
    


# Draw the tree
# Plot graphs, loss, etc. (separate class for this, can be reused)