import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def plotDecisionBoundary(clf, X, y, points, title):
	plt.clf()

	# step size in the mesh
	h = .02

	# Create a mesh of points
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Classify over the mesh points
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Plot the result into a coloured filled contour plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	plt.colorbar()

	# Plot the training points for reference
	plt.scatter(points[:, 0], points[:, 1], c=points[:,2], cmap=plt.cm.prism)
	plt.xlabel('X0')
	plt.ylabel('X1')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(title)

	plt.show()

def plotDecisionBoundaryAtAxis(fig, axis, clf, X, y, points, title):
	# step size in the mesh
	h = .02

	# Classify over the mesh points
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Classify over the mesh points
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Plot the result into a coloured filled contour plot
	Z = Z.reshape(xx.shape)
	cf = axis.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	fig.colorbar(cf, ax=axis)

	# Plot the training points for reference
	axis.scatter(points[:, 0], points[:, 1], c=points[:,2], cmap=plt.cm.prism, alpha=0.5)
	axis.set_xlabel('X0')
	axis.set_ylabel('X1')
	axis.set_xlim(xx.min(), xx.max())
	axis.set_ylim(yy.min(), yy.max())
	axis.set_xticks(())
	axis.set_yticks(())
	axis.text(0.5, 0.5, title, horizontalalignment='left', transform=axis.transAxes)

def plotDataSet(X,y):
	# Plot data
	plt.clf()
	plt.title("SVM data")
	plt.scatter(X[:,0], X[:,1], c=y, marker='o',  cmap=plt.cm.prism)
	plt.show()

def plotErrors(index, training_error, cv_error, title):
	plt.clf()
	plt.plot(index, training_error, "-b", label = "Training error", linewidth=2)
	plt.plot(index, cv_error, "-r" , label = "CV error", linewidth=2)
	plt.xlabel('X0')
	plt.ylabel('X1')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode="expand", borderaxespad=0.)
	plt.title(title)
	plt.show()

def SVMvsC():
	X, y = datasets.make_moons(2000, noise=0.7, random_state=1)
	X_train = X[:1000,:]
	y_train = y[:1000]
	X_cv = X[1000:,:]
	y_cv = y[1000:]
	points = np.c_[X,y]

	c = np.linspace(0.001, 1.0, num=1000)
	training_errors = np.zeros(c.shape)
	cv_errors = np.zeros(c.shape)
	i=0
	for _c in c:
		clf = SVC(C=_c, kernel='rbf', gamma=1.0)
		clf.fit(X_train,y_train)
		training_errors[i] = 1.0-clf.score(X_train, y_train)
		cv_errors[i] = 1.0-clf.score(X_cv, y_cv)
		i+=1

	plotErrors(c, training_errors, cv_errors, "")

def SVMvsTrainingSetSize():
	X, y = datasets.make_moons(2000, noise=0.7, random_state=1)
	points = np.c_[X,y]

	m = np.arange(10,1000)
	training_errors = np.zeros(m.shape)
	cv_errors = np.zeros(m.shape)
	X_cv = X[1000:,:]
	y_cv = y[1000:]

	i=0
	for _m in m:
		X_train = X[:_m,:]
		y_train = y[:_m]
		clf = SVC(C=1.0, kernel='rbf', gamma=1.0)
		clf.fit(X_train, y_train)
		training_errors[i] = 1.0-clf.score(X_train, y_train)
		cv_errors[i] = 1.0-clf.score(X_cv, y_cv)
		i+=1

	plotErrors(m, training_errors, cv_errors, "")

def SingleTest():
	X, y = datasets.make_moons(2000, noise=0.7, random_state=1)
	points = np.c_[X,y]
	clf = SVC(C=1.0, kernel='rbf', gamma=1.0)
	clf.fit(X, y)
	plotDecisionBoundary(clf, X, y, points, "Gaussain kernel SVM example")

def DecisionBoundariesOverTrainingSet():
	X, y = datasets.make_moons(2000, noise=0.2, random_state=1)
	points = np.c_[X,y]
	X, y = shuffle(X,y)
	sizes = np.array([30, 100, 200, 300, 400, 800])

	fig, axes = plt.subplots(3,2)
	fig.suptitle("Gaussian kernel SVM decision boundaries")
	for _size, ax in zip(sizes, axes.ravel()):
		X_slice = X[:_size,:]
		y_slice = y[:_size]
		clf = SVC(1.0, kernel='rbf', gamma=1.0)
		clf.fit (X_slice, y_slice)
		plotDecisionBoundaryAtAxis(fig, ax, clf, X_slice, y_slice, points, str(_size))

	fig.tight_layout()
	plt.show()

#DecisionBoundariesOverTrainingSet()

#SVMvsTrainingSetSize()

SingleTest()

#SVMvsC()