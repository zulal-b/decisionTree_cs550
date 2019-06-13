'''

Decision Tree Classifier
Part 1 : Own Implementation
March, 2018

Zulal Bingol

'''

from __future__ import division
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import sys
import numpy as np
import math

left_index = 0
right_index = 1

class dec_node:
	def __init__(self, grouped_samples, grouped_labels, rule, pikachu, left, right, depth, gini, class_def=None):
		self.grouped_samples = grouped_samples
		self.grouped_labels = grouped_labels
		self.rule = rule
		self.pikachu = pikachu
		self.left = left
		self.right = right
		self.depth = depth
		self.gini = gini
		self.class_def = class_def

	def setClass(self, defined_class):
		self.class_def = defined_class

def calc_gini_index(grouped_labels, num_samples, unique_labels):

	gini = 0
	for group in grouped_labels:
		group_size = len(group)
		
		if group_size == 0:
			break
		
		current_score = 0
		for u in unique_labels:
			current_sum_ratio = group.count(u) / group_size
			#print "---", u, "--", current_sum_ratio
			current_score += current_sum_ratio ** 2
		
		gini += (1 - current_score) * (group_size / num_samples)   

	return gini

def decide_rule(feature_values, samples, labels, feature_index, num_samples, unique_labels):

	coef = {0.2, 0.4, 0.6, 0.8}
	indices = [int(x * len(feature_values)) for x in coef]
	
	inner_ginis = []
	for i in indices:
		grouped_sample, grouped_labels = split_binary(samples, labels, feature_values[i], feature_index)
		#print "feat ", i, " == ",  current_rule,"\t", len(grouped_sample[0]), len(grouped_labels[1])

		current_gini = calc_gini_index(grouped_labels, num_samples, unique_labels)
		inner_ginis.append(current_gini)

	best_gini = min(inner_ginis)
	best_rule_ind = inner_ginis.index(best_gini)
	best_rule = feature_values[indices[best_rule_ind]]

	return best_gini, best_rule

def split_parent(samples, labels, num_samples, num_features, unique_labels, current_depth):
	
	gini_scores = []
	current_rules = []
	for i in range(0, num_features):
		feature_values = [float(samples[j][i]) for j in range(0, num_samples)]
		
		current_gini, current_rule = decide_rule(feature_values, samples, labels, i, num_samples, unique_labels)
		current_rules.append(current_rule)
		gini_scores.append(current_gini)

	best_gini = min(gini_scores)
	#print "BEST GINI", best_gini
	chosen_pikachu = gini_scores.index(best_gini)
	#print "chosen_pikachu = ", chosen_pikachu

	grouped_sample, grouped_labels = split_binary(samples, labels, current_rules[chosen_pikachu], chosen_pikachu)
	root = dec_node(grouped_sample, grouped_labels, current_rules[chosen_pikachu], chosen_pikachu, None, None, current_depth, best_gini)

	del(samples)

	return root

def split_binary(parent_list, labels, rule, feature_index):

	discrete_features = [i for i in range(1, 16)]

	if feature_index in discrete_features:
		left_ind = [i for i in range(0, len(parent_list)) if float(parent_list[i][feature_index]) == 0]
	else:
		left_ind = [i for i in range(0, len(parent_list)) if float(parent_list[i][feature_index]) <= rule]

	left = [] 
	right = [] 
	left_labels = [] 
	right_labels = []
	for i in range(0 , len(parent_list)):
		if (i in left_ind):
			left.append(parent_list[i])
			left_labels.append(labels[i])
		else:
			right.append(parent_list[i])
			right_labels.append(labels[i])

	grouped_sample = [left, right]
	grouped_labels = [left_labels, right_labels]

	return grouped_sample, grouped_labels

def continue_training(file, parent, num_features, unique_labels, max_depth, current_depth):

	#print "-----------------------------CURRENT DEPTH = ", current_depth

	if (float(parent.depth) < float(max_depth) and len(parent.grouped_samples[left_index]) != 0 and parent.gini != 0 and parent.class_def == None):
		left = split_parent(parent.grouped_samples[left_index], parent.grouped_labels[left_index], len(parent.grouped_samples[left_index]), num_features, unique_labels, current_depth + 1)
		parent.left = left
		print "----INTO LEFT == SIZE for left", len(parent.grouped_samples[left_index])
		print "------------------------left depth = ", left.depth
		print "left gini", left.gini
		print "left rule", left.pikachu
		file.write("----INTO LEFT == SIZE for left %d\n" % len(parent.grouped_samples[left_index]))
		file.write("------------------------left depth = %d\n" % left.depth)
		file.write("left gini %f\n" % left.gini)
		file.write("left rule %f\n" % left.pikachu)

		continue_training(file, left, num_features, unique_labels, max_depth, left.depth)
		
	elif (float(parent.depth) >= float(max_depth) or len(parent.grouped_samples[left_index]) == 0 or parent.gini == 0) and (parent.class_def == None):
		terminate(file, parent)

	if(float(parent.depth) < float(max_depth) and len(parent.grouped_samples[right_index]) != 0 and parent.gini != 0 and parent.class_def == None):
		right = split_parent(parent.grouped_samples[right_index], parent.grouped_labels[right_index], len(parent.grouped_samples[right_index]), num_features, unique_labels, current_depth + 1)
		parent.right = right
		
		print "----INTO RIGHT == SIZE for right", len(parent.grouped_samples[right_index])
		print "------------------------right depth = ", right.depth
		print "right gini", right.gini
		print "right rule", right.pikachu
		file.write("----INTO RIGHT == SIZE for right %d\n" % len(parent.grouped_samples[right_index]))
		file.write("------------------------right depth = %d\n" % right.depth)
		file.write("right gini %f\n" % right.gini)
		file.write("right rule %f\n" % right.pikachu)
		
		continue_training(file, right, num_features, unique_labels, max_depth, right.depth)
	
	elif(float(parent.depth) >= float(max_depth) or len(parent.grouped_samples[right_index]) == 0 or parent.gini == 0) and (parent.class_def == None):
		terminate(file, parent) #////// change this

	print "END OF ROUND"

	return

def terminate(file, node):
 	
 	if (len(node.grouped_labels) == 0):
 		print "Error: terminate"
 		sys.exit()

 	labels = [group[i] for group in node.grouped_labels for i in range(0, len(group))]
 	max_class = max(set(labels), key=labels.count)
 	node.setClass(max_class)
 	
 	print "Total: ", len(labels), "Left: ", len(node.grouped_labels[left_index]), "Right: ", len(node.grouped_labels[right_index])
 	file.write("Total: {} Left: {} Right: {} \n" .format(len(labels), len(node.grouped_labels[left_index]), len(node.grouped_labels[right_index])))
 	print "Class label = ", max_class
 	file.write("Class Label = %s\n" % max_class)
	#print "NODE :", node.pikachu, sign, node.rule 

	return

def train_by_gini(file, training_samples, training_labels, max_depth):

	num_samples = len(training_samples)
	label_ind = num_features = len(training_samples[0]) - 1
	
	unique_labels = []
	for i in range(0, len(training_samples)):
		if (training_samples[i][label_ind] not in unique_labels):
			unique_labels.append(training_samples[i][label_ind])

	root = split_parent(training_samples, training_labels, num_samples, num_features, unique_labels, 0)
	print "ROOT's selected feature is ", root.pikachu
	print "ROOT's rule is ", root.rule
	print "ROOT's gini is ", root.gini
	print "ROOT LEFT size", len(root.grouped_samples[left_index])
	print "ROOT RIGHT size", len(root.grouped_samples[right_index]) 
	print "----------------------------------------------------"

	file.write("ROOT's selected feature is %d \n" % root.pikachu)
	file.write("ROOT's rule is %f\n" % root.rule)
	file.write("ROOT's gini is %f\n" % root.gini)
	file.write("ROOT LEFT size %d\n" % len(root.grouped_samples[left_index]))
	file.write("ROOT RIGHT size %d\n" % len(root.grouped_samples[right_index])) 
	file.write("----------------------------------------------------\n")

	continue_training(file, root, num_features, unique_labels, max_depth, 0)
	
	return root

def test_samples(file, root, samples):

	print "Testing is starting with size ", len(samples), "..."
	file.write("Testing is starting with size %d ...\n" % len(samples))
	test_labels = []
	for i in range(0, len(samples)):
		iter_node = root

		while(iter_node.class_def == None):

			rule_index = iter_node.pikachu
			rule = iter_node.rule

			if (float(samples[i][rule_index]) <= float(rule)):
				iter_node = iter_node.left

			elif (float(samples[i][rule_index]) > float(rule)):
				iter_node = iter_node.right

			else:
				print "Error: test_samples"
				sys.exit()

		sample_label = iter_node.class_def
		test_labels.append(sample_label)

	#print test_labels
	return test_labels

def getResults(file, true_labels, predicted_labels):

	accuracy_testing = accuracy_score(true_labels, predicted_labels)
	f1_s = f1_score(true_labels, predicted_labels, average='weighted')
	confusion_mat = confusion_matrix(true_labels, predicted_labels)
	
	print "RESULTS: "
	print "Accuracy = ", accuracy_testing
	print "F1 score = ", f1_s
	print "Confusion matrix:"
	print confusion_mat
	print "--------------------------------------"

	file.write("RESULTS: \n")
	file.write("Testing accuracy, f1_score = {} , {} \n".format(accuracy_testing, f1_s))
	file.write("Confusion_matrix: \n")
	file.write(confusion_mat)
	file.write("\n--------------------------------------\n")
	
	return accuracy_testing, f1_s

def readFile(file):
	
	samples = []
	labels = []

	with open(file, 'r') as f:

		for line in f:
			values = tuple(line.split(" "))
			sample = list(values)
			sample.remove('')
			sample.remove('\n')
			label_ind = len(sample) - 1
			
			labels.append(sample[label_ind])
			samples.append(sample)

	f.close()

	return samples, labels

def apply_k_fold(file, training_samples, training_labels, max_depth):

	kf = KFold(n_splits=3)
		
	inner_accuracies = []
	inner_f1_scores = []
	inner_temp_roots = []
	file.write("K-FOLD MODE: for depth = %d----------------------------------------------------\n\n" % max_depth)
	
	for train_index, test_index in kf.split(training_samples):
		
		x_train = [training_samples[i] for i in train_index]
		x_train_labels = [training_labels[i] for i in train_index]
		x_test = [training_samples[j] for j in test_index]
		x_test_labels = [training_labels[j] for j in test_index]
		temp_root = train_by_gini(file, x_train, x_train_labels, max_depth)

		predicted_labels_x_test = test_samples(file, temp_root, x_test)
		temp_accuracy, temp_f1_score = getResults(file, x_test_labels, predicted_labels_x_test)
		
		inner_accuracies.append(temp_accuracy)
		inner_f1_scores.append(temp_f1_score)
		inner_temp_roots.append(temp_root)

	max_f1_score = max(inner_f1_scores)
	ind = inner_f1_scores.index(max_f1_score)
	chosen_root = inner_temp_roots[ind]

	file.write("Accuracies = ")
	for item in inner_accuracies:
		file.write("%f " % item)
	file.write("\n")
	print "f1_scores: "
	file.write("F1_Scores = ")
	for item in inner_f1_scores:
		file.write("%f " % item)
		print item
	file.write("\nK-FOLD END for depth = %d -----------------------------------------------\n\n" % max_depth)

	return chosen_root, max_f1_score

def plotEmAll(depth_boundary, training_accuracies, testing_accuracies, under_training_accuracies, under_testing_accuracies, filename):

	x_axis = [i for i in range (1, depth_boundary)]
	plt.subplot(2, 1, 1)
	training_plot, = plt.plot(x_axis, training_accuracies, 'b-o', label='Training')
	testing_plot, = plt.plot(x_axis, testing_accuracies, 'r-o', label='Testing')
	plt.title('Raw vs Undersampled')
	plt.legend(handles=[training_plot, testing_plot])
	plt.ylabel('Raw')

	plt.subplot(2, 1, 2)
	plt.plot(x_axis, under_training_accuracies, 'b-o')
	plt.plot(x_axis, under_testing_accuracies, 'r-o')
	plt.xlabel('Max Depth')
	plt.ylabel('Undersampled')
 
	plt.savefig(filename)
	plt.show()

	return

def checkOverfit(root):

	if (root.class_def != None):
		if (float(root.gini) != 0):
			return False

	if (root.left != None):
		checkOverfit(root.left)
	if (root.right != None):
		checkOverfit(root.right)

	return True

def printRecurse(root, depth, side):
	if root is None:
		return
	if depth == 0:
		if (side == left_index):
			print "LEFT NODE at depth ", root.depth
			print "X_", root.pikachu, " <= ", root.rule
			print "Gini Index = ", root.gini
			if root.class_def == None:
				print "Left: ", len(root.grouped_samples[left_index]), "Right: ", len(root.grouped_samples[right_index])
			else:
				print "\tClass = ", root.class_def
		elif (side == right_index):
			print "RIGHT NODE at depth ", root.depth
			print "X_", root.pikachu, " <= ", root.rule
			print "Gini Index = ", root.gini
			if root.class_def == None:
				print "Left: ", len(root.grouped_samples[left_index]), "Right: ", len(root.grouped_samples[right_index])
			else:
				print "\tClass = ", root.class_def
		else:
			print "ROOT NODE at depth 0"
			print "X_", root.pikachu, " <= ", root.rule
			print "Gini Index = ", root.gini
			if root.class_def == None:
				print "Left: ", len(root.grouped_samples[left_index]), "Right: ", len(root.grouped_samples[right_index])
			else:
				print "\tClass = ", root.class_def
		print "\n"
	elif depth > 0:
		printRecurse(root.left, depth-1, left_index)
		printRecurse(root.right, depth-1, right_index)

def printTreeBFS(root, depth):

	print "\nDecision Tree :"
	print "============================================"
	for i in range(0, depth+1):
		printRecurse(root, i, -1)


def main():

	training_file = sys.argv[1]
	testing_file = sys.argv[2]
	#max_depth = sys.argv[3]
	outputFile = "decision_tree.txt"
	k_fold_file = "k_fold_analysis.txt"
	training_accuracies = []
	testing_accuracies = []
	training_f1_scores = []
	testing_f1_scores = []
	depth_boundary = 10

	training_samples, training_labels = readFile(training_file)
	testing_samples, testing_labels = readFile(testing_file)

	print "Prepocessing... DONE"
	outputFile = open(outputFile, "w")
	k_fold_file = open(k_fold_file, "w")
	training_size = len(training_samples)
	temp_roots = []
	overfits = []

	for max_depth in range(1, depth_boundary):

		outputFile.write("TRAINING WITH MAX_DEPTH %d =================================\n" % max_depth)

		temp_root, f1_s = apply_k_fold(k_fold_file, training_samples, training_labels, max_depth)

		if (checkOverfit(temp_root) == False):
			k_fold_file.write("OVERFIT AT %d DEPTH, results will be discarded...\n" % max_depth)
			print "\nOVERFIT AT ", max_depth, " DEPTH, results will be discarded...\n"
			overfits.append(max_depth)
		temp_roots.append(temp_root)
		training_f1_scores.append(f1_s)

		'''
		# Testing with Training Data
		outputFile.write("Testing with training data...\n")
		predicted_labels_train = test_samples(outputFile, root, training_samples)
		training_accuracy, training_f1_score = getResults(outputFile, training_labels, predicted_labels_train)
		training_accuracies.append(training_accuracy)
		training_f1_scores.append(training_f1_score)'''

	max_f1_score = max(training_f1_scores)
	ind = training_f1_scores.index(max_f1_score)
	opt_depth = ind+1
	root = temp_roots[ind]
	print "OVERFITS", overfits

	printTreeBFS(root, opt_depth)

	#Testing with Testing Data
	outputFile.write("Optimum depth is %d \n" % opt_depth)
	outputFile.write("Testing with testing data...\n")
	print "Optimum depth is ", opt_depth
	print "Testing with testing data..............................................................................................."
	predicted_labels_test = test_samples(outputFile, root, testing_samples)
	testing_accuracy, testing_f1_score = getResults(outputFile, testing_labels, predicted_labels_test)
	#testing_accuracies.append(testing_accuracy)
	#testing_f1_scores.append(testing_f1_score)

	print "======================================UNDERSAMPLING===================================="
	outputFile.write("======================================UNDERSAMPLING====================================\n")
	k_fold_file.write("======================================UNDERSAMPLING====================================\n")

	new_training_samples = []
	new_training_labels = []
	max_count = 100
	current_count = 0
	label_ind = len(training_samples[0])-1 
	
	for v in training_samples:
		
		if (float(v[label_ind]) == 3 and current_count > max_count):
			continue
		
		else:
			new_training_samples.append(v)
			new_training_labels.append(v[label_ind])
			if (float(v[label_ind]) == 3):
				current_count += 1

	under_training_accuracies = []
	under_training_f1_scores = []
	under_testing_accuracies = []
	under_testing_f1_scores = []
	temp_roots = []
	overfits = []
	
	for max_depth in range(1, depth_boundary):

		outputFile.write("TRAINING WITH MAX_DEPTH %d =================================\n" % max_depth)

		temp_root, f1_s = apply_k_fold(k_fold_file, new_training_samples, new_training_labels, max_depth)

		if (checkOverfit(temp_root) == False):
			k_fold_file.write("OVERFIT AT %d DEPTH, results will be discarded...\n" % max_depth)
			print "\nOVERFIT AT ", max_depth, " DEPTH, results will be discarded...\n"
			overfits.append(max_depth)
		temp_roots.append(temp_root)
		under_training_f1_scores.append(f1_s)

	max_f1_score = max(under_training_f1_scores)
	ind = under_training_f1_scores.index(max_f1_score)
	opt_depth = ind+1
	new_root = temp_roots[ind]
	print "OVERFITS", overfits

	printTreeBFS(new_root, opt_depth)

	#Testing with Testing Data
	outputFile.write("Optimum depth is %d \n" % opt_depth)
	outputFile.write("Testing with testing data...\n")
	print "Optimum depth is ", opt_depth

	#Testing with Testing Data
	predicted_labels_test = test_samples(outputFile, new_root, testing_samples)
	testing_accuracy, testing_f1_score = getResults(outputFile, testing_labels, predicted_labels_test)
	#under_testing_accuracies.append(testing_accuracy)
	#under_testing_f1_scores.append(testing_f1_score)

	#plotEmAll(depth_boundary, training_accuracies, testing_accuracies, under_training_accuracies, under_testing_accuracies, "Accuracy_Comparisons")
	#plotEmAll(depth_boundary, training_f1_scores, testing_f1_scores, under_training_f1_scores, under_testing_f1_scores, "F1_scores_Comparison")

	k_fold_file.close()
	outputFile.close()
	print "Peace out !"


if __name__ == "__main__":
    main()