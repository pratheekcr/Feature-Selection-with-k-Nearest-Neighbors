from scipy import stats
import math
import operator
import time
import sys

# Import Dataset
def getDataset(dataset_file):
	with open(dataset_file) as file:
		data = file.readlines()
		dataset = []
		for line in data:
			try:
				numbers = line.lstrip(" ")
				numbers = [float(number) for number in numbers.split()]
				numbers[0] = int(numbers[0])
				dataset.append(numbers)
			except ValueError:
				print("Error occured on line"+line)
	return dataset

# Normalize the Instances between 1 and -1
def normalize(activeDataSet):
	new_data = stats.zscore(activeDataSet)
	return new_data

# Find the Eucledian Distance
def eucledianDistance(active,test,train): 	
	distance= 0
	for i in range (len(active)):
		if active[i]:
			distance += pow((test[i]-train[i]),2)
	return math.sqrt(distance)

# Leave-one Out Cross Validation
def leaveOneOutValidation(active, trainingSet, testInstance):
	knn=1
	dis = []
	for x in range(len(trainingSet)):
         dist = eucledianDistance(active, testInstance, trainingSet[x])
         dis.append((trainingSet[x], dist))
	dis.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(knn):
		neighbors.append(dis[x][0])
	return neighbors

# Calculate accuracy
def knnAccuracy(data,activation):
	accuracy = 0.00
	for i in range(len(data)):
		instancesRemain = list(data)
		leaveoneInstance = instancesRemain.pop(i)
		neighbors = leaveOneOutValidation(activation, instancesRemain,leaveoneInstance)
		if (len(neighbors) == 1):
			for x in range(len(neighbors)):
				if (neighbors[x][0] == leaveoneInstance[0]):
					accuracy += 1
	knn = (accuracy/len(data))* 100
	return knn

def forwardSelection(data,features):
	activation = [1] * len(data[0])
	activation[0] = 0
	localAccuracy = knnAccuracy(data,activation)
	print "Running nearest neighbor with all ",features," features, using \"leaving-one-out\" evaluation, I get an accuracy of ",localAccuracy,"%\n"
	print "Beginning search.\n"
	start=time.time()
	posFlags = [i for i in range(1,features+1)]
	featureSet = []
	bestFeatureSet = []
	globalAccuracy = 0.0
	for i in range(features):
		retValue  = forwardSubsets(data,posFlags,featureSet,globalAccuracy)
		featureSet =retValue[0]
		accuracy = retValue[1]
		if (accuracy > globalAccuracy):
			globalAccuracy = accuracy
			bestFeatureSet = list(featureSet)
	end=time.time()
	total=end-start
	print "Finished search!! The best feature subset is {",','.join(str(i) for i in bestFeatureSet),"},with ",globalAccuracy,"%""Accuracy"
	print("total time elapsed",total)

def forwardSubsets(dataSet,possibleFlags,currFeatures, bestAccuracy):
	feature=0
	accuracy = 0.0
	featuresRemain = [i for i in possibleFlags if i not in currFeatures]
	featureAccuracy = [0.0]* len(featuresRemain)
	for i in featuresRemain:
		flags = [0]* (len(possibleFlags)+1)
		flags[i] = 1
		for j in currFeatures:
			flags[j] = 1
		accuracy = knnAccuracy(dataSet,flags)
		featureAccuracy[feature] = accuracy
		featureSet = list(currFeatures)
		featureSet.append(i)
		print "Using feature(s) {", ','.join(str(i) for i in featureSet),"} accuracy is ",featureAccuracy[feature],"%"
		feature += 1
	featureSet = list(currFeatures)
	maximumAccuracy= max(featureAccuracy)
	for i in range(len(featuresRemain)):
		if (featureAccuracy[i]== maximumAccuracy):
			featureSet.append(featuresRemain.pop(i))
			break
	print "\n"
	if(maximumAccuracy < bestAccuracy):
		print "(Warning, Accuracy has decreased!",
		print " Continuing search in case of local maxima)",
	print "Feature set{",','.join(str(i) for i in featureSet),"} was best, accuracy is ", maximumAccuracy,"%\n"
	return (featureSet, maximumAccuracy)

def backwardElimination(data,features):
	flags = [1] * len(data[0])
	flags[0] = 0
	accuracy = knnAccuracy(data,flags)
	print "Running nearest neighbor with all ",features," features, using \"leaving-one-out\" evaluation, I get an accuracy of ",accuracy,"%\n"
	print "Beginning search.\n"
	start = time.time()
	posFlags = [i for i in range(1,features+1)]
	featureSet = [i for i in range(1,features+1)]
	bestFeatureSet = [i for i in range(1,features+1)]
	bestAccuracy = 0.0
	for i in range(features-1):
		retValue  = backwardSubsets(data,posFlags,featureSet,bestAccuracy)
		featureSet =retValue[0]
		accuracy = retValue[1]
		if (accuracy > bestAccuracy):
			bestAccuracy = accuracy
			bestFeatureSet = list(featureSet)
	end = time.time()
	total = end - start
	print "Finished search!! The best feature subset is {",
	print ','.join(str(i) for i in bestFeatureSet),
	print "},which has an accuracy of ",bestAccuracy,"%"
	print("total time elapsed", total)

def backwardSubsets(data,possibleFlags,currFeatures, bestAccuracy):
	feature = 0
	accuracy = 0.0
	featuresRemain = list(currFeatures)
	featureAccuracy = [0.0]* len(featuresRemain)
	for i in featuresRemain :
		flags = [0]* (len(possibleFlags)+1)
		for j in currFeatures:
			flags[j] = 1
		flags[i] = 0
		accuracy = knnAccuracy(data,flags)
		featureAccuracy[feature] = accuracy
		featureSet = list(currFeatures)
		featureSet.remove(i)
		print "Using feature(s) {",
		if (len(featureSet)== 1):
			print featureSet[0],
		else:
			print ','.join(str(i) for i in featureSet),
		print"} accuracy is ",featureAccuracy[feature],"%"
		feature += 1
	featureSet = list(currFeatures)
	maximumAccuracy = max(featureAccuracy)
	weakestLink = 0
	for i in range(len(featureAccuracy)):
		if (featureAccuracy[i]== maximumAccuracy):
			weakestLink = featuresRemain.pop(i)
			featureSet.remove(weakestLink)
			break
	if(maximumAccuracy < bestAccuracy):
		print "\n(Warning, Accuracy has decreased!",
	print "\nFeature set{",','.join(str(i) for i in featureSet),"} was best, removing",weakestLink," has the highest accuracy,", maximumAccuracy,"%\n"
	return (featureSet, maximumAccuracy)

def pratheek(data,features): #Backward Elimination with pruning
	active = [1] * len(data[0])
	active[0] = 0
	accuracy = knnAccuracy(data, active)
	print "Running nearest neighbor with all " , features, " features, using \"leaving-one-out\" evaluation, I get an accuracy of " , accuracy, "%\n"
	print "Beginning search.\n"
	start = time.time()
	posFlags = [i for i in range(1, features + 1)]
	featureSet = [i for i in range(1, features + 1)]
	bestFeatureSet = [i for i in range(1, features + 1)]
	globalAccuracy = 0.0
	for i in range(features - 1):
		retValue = backwardSubsets(data, posFlags, featureSet, globalAccuracy)
		featureSet = retValue[0]
		localAccuracy = retValue[1]
		if (localAccuracy < globalAccuracy):
			end = time.time()
			total = end - start
			print("total time elapsed", total)
			print"Finished search!! The best feature subset is {", ','.join(str(i) for i in bestFeatureSet), "},which has an accuracy of ", globalAccuracy, "%"
			sys.exit()
		if (localAccuracy > globalAccuracy):
			globalAccuracy = localAccuracy
			bestFeatureSet = list(featureSet)

	print"Finished search!! The best feature subset is {", ','.join(str(i) for i in bestFeatureSet), "},which has an accuracy of ", globalAccuracy, "%"

# Main
def main():
	print ("Welcome to Pratheek's Feature Selection Algorithm")
	testset = raw_input('Select the Dataset file:')
	dataset=getDataset(testset)
	print("Please wait while I normalize the data!")
	new_data=normalize(dataset)
	print("Data Normalized.")
	instances = len(new_data)
	features = len(new_data[0]) - 1
	print "This dataset has ", features, " features and " \
		, instances, " instances.\n"
	print "1) Forward Selection"
	print "2) Backward Elimination"
	print "3) Pratheek's Algorithm"
	choice = int(input("Select the Algorithm:"))
	if (choice == 1):
		forwardSelection(new_data,features)
	elif (choice == 2):
		backwardElimination(new_data,features)
	elif (choice == 3):
		pratheek(new_data,features)
	else:
		print("The choice is not correct. Please select a valid choice")

if __name__ =='__main__':
	main()