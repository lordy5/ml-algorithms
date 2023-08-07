import numpy as np

# Classification decision tree
class Tree:
    def __init__(self, X_train, y_train, max_depth, min_samples):
        self.X_train = X_train
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.root = Node(0, X_train, y_train)
    
    def build_tree(self, node):
        if node.depth > self.max_depth or node.samples < self.min_samples:
            '''Criteria for splitting is no longer met, so create a leaf node in the tree.
            To do this, we find the majority class label in the node's data, and assign that
            label to the node.'''
            node.set_class()
        else:
            #split data and create child nodes
            left_node, right_node = node.split()
            
            # When splitting, if a node is pure, it's left and right children are set to itself
            # so that we can stop recursion for that branch
            if left_node == node and right_node == node:
                return
            self.build_tree(left_node)
            self.build_tree(right_node)

    # Calls the recursive predict function to predict the label for this datapoint
    def make_prediction(self, x):
        return self.root.predict(x)
 
class Node:
    def __init__(self, depth, features, labels):
        self.features = features
        self.labels = labels
        self.left = None
        self.right = None

        self.depth = depth
        self.samples = np.shape(features)[0]
        self.split_feature = None
        self.leaf_label = None
        self.is_leaf = False
        self.threshold_value = 0

    # splits data, returns resulting left and right child nodes
    def split(self):
        self.left, self.right = self.find_and_split()
        return self.left, self.right
    
    # If node is a leaf, find and set the class
    def set_class(self):
        # Find the label with the most non-zero entries (Assuming each label has a column,
        # where a 1 is true and 0 is false for that class)
        nonzero_counts = np.count_nonzero(self.labels, axis=0)
        self.leaf_label = np.argmax(nonzero_counts)
        self.is_leaf = True

    def gini_impurity(self, labels):
        num_labels = np.shape(labels)[1]
        num_samples = np.shape(labels)[0]
        sum = 0

        for label in range(num_labels):
            sum += (np.count_nonzero(labels[:, label]) / num_samples)**2

        return 1 - sum

    # Finds best feature and threshold to split at, returns the threshold value
    def find_and_split(self):        
        total_samples = np.shape(self.features)[0]
        num_features = np.shape(self.features)[1]
        best_gain = 0
        best_feature = None
        best_left_features, best_right_features = None, None
        best_left_labels, best_right_labels = None, None
        threshold = 0
        left_child = None
        right_child = None

        impurity = self.gini_impurity(self.labels)

        # If node is completely pure, it should be a leaf
        if impurity == 0:
            self.set_class()
            return self, self

        for feature in range(num_features):
            # Sort labels by the current feature, to help with determining the threshold value
            sorted_indices = np.argsort(self.features[:, feature])
            sorted_labels = self.labels[sorted_indices]
            sorted_features = self.features[sorted_indices]

            for row in range(total_samples - 1):
                left_labels = sorted_labels[:(row + 1)]
                right_labels = sorted_labels[(row + 1):]
                left_features = sorted_features[:(row + 1)]
                right_features = sorted_features[(row + 1):]

                num_left_datapoints = np.shape(left_labels)[0]
                num_right_datapoints = np.shape(right_labels)[0]

                # gini impurity for left and right children
                left_impurity = self.gini_impurity(left_labels)
                right_impurity = self.gini_impurity(right_labels)
                gini_gain = impurity - (left_impurity * num_left_datapoints / total_samples) - \
                    (right_impurity * num_right_datapoints / total_samples)

                # Check if this gain is better than previous best
                if gini_gain > best_gain:
                    best_gain = gini_gain
                    best_feature = feature
                    best_right_features, best_left_features = right_features, left_features
                    best_right_labels, best_left_labels = right_labels, left_labels
                    threshold = sorted_features[row][feature]

        left_child = Node(self.depth + 1, best_left_features, best_left_labels)
        right_child = Node(self.depth + 1, best_right_features, best_right_labels)
        self.threshold_value = threshold

        self.split_feature = best_feature
        return left_child, right_child

    # Traverse the tree until we get to a leaf, then the predicted class is just the leaf's label
    def predict(self, x):
        if self.is_leaf:
            return self.leaf_label
        elif x[self.split_feature] <= self.threshold_value:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

        
    


# Draw the tree
# Plot graphs, loss, etc. (separate class for this, can be reused)