# Stepwise Regression with R-squared for Linear Regression in Python

I came up with no results in python distributed packages when I searched for stepwise regression which is a robust way of feature selection for linear regression. I built this stepwise regression function which first forward selects features on two criteria; F-test significance the feature adds to the model, and increased test r-squared value, then backward eliminates features if they do not cause significance decrease when dropped from the model features. If you have something to say about this, you could send an e-mail to ozgurdoganbirol@gmail.com.

## Stepwise Regression

If a prediction variable is tought to be showing linearity with its properties, then a linear regression model is applied to the dataset. Even though adding more features to the model generally improves the prediction performance, after a certain point adding more features become redundant: It does not decrease the error of the model. Besides that, adding too many features to the model might cause a very specific fit for the training data, but low prediction capability for new data, which is called overfit. For these reasons, an effective way of selecting features is to perform it one by one, this is called forward selection. First, all the features are fitted and the best feature is selected. In the next step, the selected feature is fitted new models with each of the rest of the features. The best model is selected. Therefore, this selection process continues in this pattern until all the rest of the features stop improving the model. 

Forward selection is not flawless though. The problem with forward selection is the interaction of the features with each other. When the model keeps adding features, sometimes, the previously added features might become redundant. The newly added features and their interaction with the other existing features might actually turn some of those existing features into obsolete for the model. This is why backward elimination is performed. After the forward selection adds a feature to the model, backward elimination tests usefulness of each of the existing features by each time eliminating one from the model. The features which did not significantly decrease the models significance are eliminated from the model. Their contributions to the model are negligible. 

This way forward selection and backward elimination are sequantially performed until algorithms spot neither beneficial, nor obsolete variables for the model. For an illustration of this method using an example please [click]: https://online.stat.psu.edu/stat501/lesson/10/10.2. The function I wrote decides the next best variable by F-test p value significance and test r-squared improvement margin. The backward elimination though only takes p values as a reference. The desired levels of those are given as inputs to the function. This way, the user could decide the tolerance of the selection and elimination. 

## Explanation of Code and Pseudocode


```

```
## Final Comments

