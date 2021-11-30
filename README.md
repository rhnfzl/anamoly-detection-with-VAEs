# Anomaly Detection with VAEs

#### Part of Assignment of Deep Learning Course (2IMM10) at TU/e

Variational Autoencoders (VAEs) provide a mathematically grounded framework for the unsupervised learning of latent representations. Besides interpreting VAEs as representation learning or generative modelling, we can also see them as performing (approximate) density approximation. VAEs are trained t optimise a lower bound to the (log) likelihood log p(X) of the data X, under the chosen model. So, for any point in data space, we can obtain an estimate of its likelihood under the trained model, by simply computing the loss function when passing this data point through the neural network (note that the loss function is the negative ELBO, so we need to multiply by -1 to obtain a likelihood estimation).

We can use this idea to perform unsupervised anomaly detection. Suppose we are given a dataset that describes some natural distribution (e.g. images of certain clothing items). For new test data, we then wish to detect whether it fits this distribution, or is significantly different (an anomaly). For example, given a dataset of shirts, we want to detect anomalies in a test data set that also contains some images of trousers. Typically, such a situation occurs when we have many examples of one class (e.g. shirts), but very few of others (the anomalies, e.g. trousers).

In this task, we will perform and evaluate such anomaly detection with VAEs. Given a training data set that consists of instances that we consider “normal”, we wish to detect anomalies in a test data set that contains both “normal” (but unseen) examples, as well as other examples which we consider anomalous. The idea is to train a VAE on the training data, such that it learns to represent “normal” data well. We can then compute the ELBO values for the test data, where ideally “normal” examples should obtain higher likelihood values than anomalous examples.

In this assignment, we will use FashionMNIST to simulate the anomaly detection task. We will omit one class from the training data, and consider the remaining 9 classes to be “normal”. The goal is then to identify the omitted class in the test data, by comparing the ELBO values obtained from a VAE trained on 9 classes.

### Task 1: Obtain anomaly detection dataset

a.  We will consider the “Trouser” class (with label 1) to be the anomalies, and consider the other 9 classes to be our “normal” data. We’ll train on normal data only, but we want to test on both normal and anomalous data to evaluate our anomaly detection framework.

* Load the FashionMNIST dataset.
* Remove all instances from the anomaly class from the training set.
*  Split the test set in two parts: the anomalous data (with label 1) and the normal data (all other labels).

b. To check if the split was done correctly, plot some random examples (at least 10 each) of:

* The new training set (without the anomaly class)
* The normal test set
* The anomaly test set

### Task 2: Design, implement, and train a VAE

a. Design a VAE for the FashionMNIST dataset with a suitable architecture, that should perform well on this dataset.

* Implement the VAE (with corresponding loss functions) and compile it.
* Print a summary (with .summary()) of the encoder and decoder.

b. Train the VAE on the FashionMNIST training dataset without the anomaly class. Make sure that you train long enough such that the loss is no longer going down. Make sure that the training output is printed (use the default verbose setting in .fit).

### Task 3: Inspect VAE performance

Qualitatively inspect if the VAE is trained well. The latent space plots we saw in the practical only work for 2-dimensional latent spaces, but you may need to increase the dimensionality of the latent space for good performance. Therefore we’ll make some plots that work for higher-dimensional latent spaces as well; reconstructions and random samples:

* Reconstructions: Take a random sample of normal training images (at least 10), and use the VAE to obtain their reconstructions. Plot both originals and reconstructions, on top of each other.
* Random samples: Randomly generate some images (at least 10) with the VAE; i.e. sample latent variables from the prior distribution, and decode them into data space. Plot the results.

### Task 4: Anomaly detection

a. Use the VAE to obtain density/likelihood estimations for the normal and anomalous test sets, i.e. compute the ELBO (the negative of the loss function) for all points in both test sets. Make sure to keep the scores for the normal and anomaly sets separate from each other.

b. Visualise the scores in a histogram (plt.hist()) as well as a density plot (sns.kdplot from the seaborn library). Use two different colours: green for normal data, red for anomalous data, and show both normal and anomalous scores in the same plot (use transparency to make visualisation clearer), i.e. one figure with both histograms, and one figure with both density plots.

c.  Given these likelihood scores, we can choose a threshold and classify all instances with a likelihood below the threshold as anomalies, and all instances with a likelihood above the threshold as “normal”. Different thresholds will give different True/False Positive/Negative scores. We can summarise the performance of all thresholds in an ROC curve, or a Precision Recall curve (the latter has been shown to be more suitable for imbalanced datasets, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/).

* Plot an ROC curve for your results, compute and show the Area Under Curve (AUC) score for quantitative evaluation.
* Plot a Precision-Recall curve for your results, compute and show the Area Under Curve (AUC) score for quantitative evaluation.

d. Ideally, a successful VAE for anomaly detection should represent (and thus reconstruct) normal data very well, but not anomalous data. Reconstruct some random images (at least 10 each) from the normal test set, as well as from the anomaly test set. Show the original images and their reconstructions on top of each other.

e. Give a detailed discussion of your results; does the anomaly detection perform well? Why do you think so? What could be improved? Discuss each of the results from parts (b), (c), and (d) separately.





