# Stepwise Regression with R-squared for Linear Regression in Python

I came up with no results in python distributed packages when I searched for stepwise regression which is a robust way of feature selection for linear regression. I built this stepwise regression function which first forward selects features on two criteria; F-test significance the feature adds to the model, and increased test r-squared value, then backward eliminates features if they do not cause significance decrease when dropped from the model features. If you have something to say about this, you could send an e-mail to ozgurdoganbirol@gmail.com.

## Stepwise Regression

If a prediction variable is thought to exhibit linearity with its properties, then a linear regression model is applied to the dataset. Even though adding more features to the model generally improves the prediction performance, after a certain point adding more features become redundant: It does not decrease the error of the model. Besides that, adding too many features to the model might cause a very specific fit for the training data, but low prediction capability for new data, which is called overfit. For these reasons, an effective way of selecting features is to perform it one by one, this is called forward selection. First, all the features are fitted and the best feature is selected. In the next step, the selected feature is fitted new models with each of the rest of the features. The best model is selected. Therefore, this selection process continues in this pattern until all the rest of the features stop improving the model. 

Forward selection is not flawless though. The problem with forward selection is the interaction of the features with each other. When the model keeps adding features, sometimes, the previously added features might become redundant. The newly added features and their interaction with the other existing features might actually turn some of those existing features into obsolete for the model. This is why backward elimination is performed. After the forward selection adds a feature to the model, backward elimination tests usefulness of each of the existing features by each time eliminating one from the model. The features which did not significantly decrease the models significance are eliminated from the model. Their contributions to the model are negligible. 

This way forward selection and backward elimination are sequantially performed until algorithms spot neither beneficial, nor obsolete variables for the model. For an illustration of this method using an example please [click]: https://online.stat.psu.edu/stat501/lesson/10/10.2. The function I wrote decides the next best variable by F-test p-value significance and test r-squared improvement margin. The backward elimination though only takes p values as a reference. The desired levels of those are given as inputs to the function. This way, the user could decide the tolerance of the selection and elimination. 

## Explanation of Code and Pseudocode

The function takes data, name of the target variable, test split size, number of maximum iterations, desired test r-squared contribution margin for forward selection, desired p-significance values as inputs. As long as the runtime continues, the function prints the progress to the terminal.  The first few lines of the function are interested in train test split of the data set. Then sum of the squared errors are calculated for it will be useful for F-test. After that, iteration variables are declared in their default or null states. Then a while loop begins and wraps everything.  This while loop breaks only if only maximum number of iterations which was given as input is reached or forward selection could not identify a contributing feature anymore. Inside this while loop, a for loop begins to perform forward selection. Algorithm fits linear models with necessary train and test sets. Their significances are calculated by SSR and MSE indicators. The following if else block evaluates whether the best feature found has a significant p-value and if it increases the r-squared value of the model. If these conditions are sustained, the train and test datasets are updated accordingly with the new features of the model. If the desired improvements were not sustained, the outer while loop breaks and the function is completed. Otherwise, at this point, the forward selection is completed. Next for loop performs the backward elimination. If the eliminated feature did not significantly decrease the P significance of the model, then it will be added to the ineffective features array.  Afterwards, all the ineffective features will be eliminated from the model and the test and train data sets are updated accordingly. These loops go on sequentially until the completion conditions (maximum iterations, no more model improvements) are in place.

```
def stepwiseRegressionRsquared(data,target_var,test_size,max_iter,contribution_margin,p_significance):
  
  ***Prepare test and train out of data. Declare the models will be linear regression. Calculate sum of squared errors for use in F-test.***
  
  ***Initiate the loop variables with default or null states.***
  
  while(number_of_non_model_features>0):    
    if number_of_current_iteration >= max_iter:
      break
      
    number_of_iterations += 1
    
    for i in non_model_features:
      x_train=existing_model_features+data[i]
      y_train=existing_model_features+data[i]
      model.fit(x_train, y_train)
      ypred_train = mdl.predict(x_train)
      ypred_test = mdl.predict(x_test)
     
      mse = mean_squared_error(y_train, ypred_train)
      ssr_increase=ssr-previous_ssr
      f_value=ssr_increase/mse
      p = 1-scipy.stats.f.cdf(f_value, 1, degrees_of_freedom)
      
      results_features.append(results) 
      sort_results_by(R2_test)
  
    if (best_result(p)<p_significance) and (best_result(R2) >= initial_R2_test + contribution_margin):
      
      ***If the model best model of the loop came up with is acceptable, update all test and train data and
      loop  variables accordingly.***
     
    else:
      break
    
    if number_of_existing_model_features>1:
      
      for i in existing_model_features:
         x_train=existing_model_features-data[existing_model_features[i]]
         y_train=existing_model_features-data[existing_model_features[i]]
         model.fit(x_train, y_train)
         ypred_train = mdl.predict(x_train)
         ypred_test = mdl.predict(x_test)
         
         ssr_decrease=previous_ssr-ssr
         f_value=ssr_decrease/mse
         p = 1-scipy.stats.f.cdf(f_value, 1, degrees_of_freedom)
         
         if p>p_significance:
          ineffective_features.append(i)
         
      if number_of_ineffective_features>0:
          
         ***If there are ineffective features, update all test and train data and
         loop  variables accordingly.***
     
     
```
## Final Comments

Even though the feature selection phase is performed both by p_significance and r-squared contribution_margin inputs, beware that all the models the forward selection loop has came up with are sorted by r-squared test contribution and only after that the top model is examined for p and r-squared test. This means this function is already making r-squared its focus. Usually, r-squared test and p significance are correlated. Despite the model decided to inspect the selection model by r-squared test, selection of a more relaxed r-squared contribution margin than p significance will not make the model more relaxed than it is because the p-significance will not select and eliminate the r-squared relaxed features anyway. So we can conclude that r-squared contribution margin is a tool of limiting the p-significance. It could only work as a more strict criterion than p-significance and not the vice-versa.
