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
            # figure out the recursion for this
            self.build_tree(left_node)
            self.build_tree(right_node)

    # Calls the recursive predict function to predict the label for this datapoint
    def make_prediction(self, x):
        self.root.predict(x)



    # Returns column number for best feature to split on
    def choose_split_feature(self):
        return 1
    
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
        # Use Gini impurity to determine the feature to split at
        
        '''Calculate Gini impurity at the node: 1 - sum(p_i^2) for 1 <= i <= k, where there
        are k classes and p_i is the probability of a sample belonging to class i at the node.'''
        
        # Calculate impurity of the node
        summation = 0
        for k in range(np.shape(self.labels)[1]):
            summation += pow((np.count_nonzero(k, axis=0) / np.shape(self.labels)[0]), 2)
        impurity = 1 - summation

        # Store gini gain of each split: tuple (feature, threshold) is key, gini value is the value
        gini_gains = {}

        # Go through each feature
        for i in range(np.shape(self.features)[1]):
            # Go through all possible splits, determining gini impurities of the left and right
            # child nodes that result from each split possibility
            for j in range(np.shape(self.features)[0]):
                features_left = self.features[:(j + 1), :]
                features_right = self.features[(j + 1):, :]
                labels_left = self.labels[:(j + 1), :]
                labels_right = self.labels[:(j + 1), :]

                # Calculate impurities
                left_summation = 0
                for k in range(np.shape(labels_left)[1]):
                    # each term in summation is the square of the proportion of data points that are of 
                    # the current label's class
                    num_left_datapoints = np.shape(labels_left)[0]
                    left_summation += pow((np.count_nonzero(k, axis=0) / num_left_datapoints), 2)

                left_impurity = 1 - left_summation
                
                right_summation = 0
                for k in range(np.shape(labels_right)[1]):
                    num_right_datapoints = np.shape(labels_right)[0]
                    right_summation += pow((np.count_nonzero(k, axis=0) / num_right_datapoints), 2)

                right_impurity = 1 - right_summation

                # Calculate gini gain, difference of the impurity of parent and 
                # weighted sum of impurities of the left and right children
                gini_gain = impurity - (num_left_datapoints * left_impurity + num_right_datapoints * right_impurity)
                gini_gains[(i, j)] = gini_gain

        # Find the highest gini gain, and return corresponding data point row to split at
        best_split = max(gini_gains, key=gini_gains.get)

        self.split_feature = best_split[0]

        return best_split[1]
    
    # Traverse the tree until we get to a leaf, then the predicted class is just the leaf's label
    def predict(self, x):
        if self.is_leaf:
            return self.leaf_label
        elif x[self.split_feature] <= self.left[self.split_feature]:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

        
    


# Draw the tree
# Plot graphs, loss, etc. (separate class for this, can be reused)