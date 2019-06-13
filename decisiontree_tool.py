'''

Decision Tree Classifier
Part 2 : sklearn.tree Decision Tree Classifier
March, 2018

Zulal Bingol

'''

from __future__ import division
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt
import sys

def applyClassification_test(classifier, sample, labels):

	predicted_labels = classifier.predict(sample)
	score = f1_score(labels, predicted_labels, average='weighted')
	confusion_mat = confusion_matrix(labels, predicted_labels)

	return score, confusion_mat

def applyClassification_train(tr_feature_values, tr_true_scores, max_height):
    
    classifier = DecisionTreeClassifier(max_depth=max_height, class_weight='balanced')
    classifier.fit(tr_feature_values, tr_true_scores)
    
    predicted_tr = classifier.predict(tr_feature_values)
    score_tr = f1_score(tr_true_scores, predicted_tr, average='weighted')
    confusion_mat = confusion_matrix(tr_true_scores, predicted_tr)
    
    return score_tr, classifier, confusion_mat

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
			del sample[label_ind]
			samples.append(sample)

	f.close()

	return samples, labels

def main():

	training_file = sys.argv[1]
	testing_file = sys.argv[2]
	outputFile = "tool_decision_tree.txt"
	k_fold_file = "tool_k_fold_analysis.txt"
	training_f1_scores = []
	testing_f1_scores = []
	depth_boundary = 10

	training_samples, training_labels = readFile(training_file)
	testing_samples, testing_labels = readFile(testing_file)

	print "Prepocessing... DONE"
	outputFile = open(outputFile, "w")
	k_fold_file = open(k_fold_file, "w")
	training_size = len(training_samples)
	classifier_bag = []
	overfits = 0

	for max_depth in range(1, depth_boundary):

		outputFile.write("TRAINING WITH MAX_DEPTH %d =================================\n" % max_depth)

		f1_s, classifier, confusion_mat_tr = applyClassification_train(training_samples, training_labels, max_depth)
		print "Training Score = ", f1_s
		
		if f1_s == 1:
			overfits = max_depth
			#break
		
		classifier_bag.append(classifier)
		training_f1_scores.append(f1_s)
		print confusion_mat_tr
		
		testing_f1_score, confusion_mat_ts = applyClassification_test(classifier, testing_samples, testing_labels)
		print "Testing Score = ", testing_f1_score
		testing_f1_scores.append(testing_f1_score)
		print confusion_mat_ts

		print "................................................"


	max_f1_score = max(training_f1_scores)
	ind = training_f1_scores.index(max_f1_score)
	opt_depth = ind+1
	selected_classifier = classifier_bag[ind]

	print "Scores = ", training_f1_scores
	print "Opt Score = ", max_f1_score
	if overfits != 0:
		print "Tree overfits at ", overfits, "depth"
		#depth_boundary = overfits

	#Testing with Testing Data
	outputFile.write("Optimum depth is %d \n" % opt_depth)
	print "Optimum depth is ", opt_depth
	
	feature_names = [i for i in range(0, 21)]
	export_graphviz(selected_classifier, out_file='tree.dot', feature_names=feature_names)

	x_axis = [i for i in range (1, depth_boundary)]
	training_plot, = plt.plot(x_axis, training_f1_scores, 'b-o', label='Training')
	testing_plot, =	plt.plot(x_axis, testing_f1_scores, 'r-o', label='Testing')
	plt.title('Training Scores vs Testing Scores')
	plt.legend(handles=[training_plot, testing_plot])
	plt.ylabel('f1_score')
	plt.xlabel('Max Depth')
	plt.savefig("Tool_result")
	plt.show()

	k_fold_file.close()
	outputFile.close()
	print "Peace out !"


if __name__ == "__main__":
    main()