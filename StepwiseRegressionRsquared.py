#Required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import scipy
from IPython.display import display

#data: Pandas dataframe, target_var: Prediction column name of data, 
#test_size: Train-test split test size, max_iter: Max. number of iterations integer, 
#contribution_margin: Feature selection criterion r-squared score increase decimal,
#p_significance: The F-test significance treshold (usually 0.05 works sensibly)
def stepwiseRegressionRsquared(data,target_var,test_size,max_iter,contribution_margin,p_significance):

    #Prepare test/train, declare linear regression as model
    X = pd.DataFrame(data).drop(axis=1,columns=[target_var])
    y = data[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    mdl = LinearRegression()

    #Calculate SST: Sum of squared errors, this will be used in F-testing of features.
    avg_y = np.mean(y_train)
    squared_errors = (y_train - avg_y) ** 2
    sst=np.sum(squared_errors)

    # Create required variables with null states or default values
    # They will provide values from iteration to iteration
    initial_feat_train = pd.DataFrame()
    initial_feat_test = pd.DataFrame()
    rest_feat_train = X_train
    rest_feat_test = pd.DataFrame()
    sorted_results = pd.DataFrame()
    initial_R2_train = 0
    initial_R2_test = 0
    number_of_iterations = 0
    p = 0

    # The loop for stepwise regression: It goes on until there is no feature left unselected
    # But it breaks if the model improves no more, or the maximum iterations has been hit.
    while len(rest_feat_train.columns) > 0:

        #The loop breaks if the maximum iterations is reached.
        if number_of_iterations >= max_iter:
            print("\nMaximum number of iterations has been reached. The latest model reported above is the best found within the maximum iteration limit.")
            mdl.fit(initial_feat_train, y_train)
            ypred_train = mdl.predict(initial_feat_train)
            ypred_test = mdl.predict(initial_feat_test)
            print("\nThe features that best explain the target:", initial_feat_train.columns.values,
                "\nThe R^2 values for training and test respectively:", r2_score(
                    y_train, ypred_train), r2_score(y_test, ypred_test),
                "\nThe coefficients of the model:", mdl.coef_,
                "\nMean squared error for the training and test respectively:", mean_squared_error(y_train, ypred_train), mean_squared_error(y_test, ypred_test))
            
            break
        else:
            pass
        
        # Let user follow the iteration number
        number_of_iterations += 1
        print("\n\n##### Number of Iterations:", number_of_iterations, "#####\n\n")



        ####First part of the code tries to add the next best feature to the model (Forward selection)####

        # Reset the results features each time before loop (The list to carry regression metrics)
        results_features = [["Features", "R^2 Train",
                            "R^2 Test", "SSR", "SSR Increase", "MSE", "F", "p"]]

        # The loop analyzes the impact of adding each feature to model seperately and selects the best
        for i in rest_feat_train.columns:
            # If the loop works for the first time
            if initial_feat_train.empty:
                # Just pick each feature seperately
                x_train = pd.DataFrame(X_train.loc[:, i])
                x_test = pd.DataFrame(X_test.loc[:, i])
            else:
                # Current model (initial_feat_) plus i'th of the rest of the features (rest_feat_)
                x_train = pd.DataFrame(
                    rest_feat_train.loc[:, i]).join(initial_feat_train)
                x_test = pd.DataFrame(
                    rest_feat_test.loc[:, i]).join(initial_feat_test)

            # Fit the model with added feature
            mdl.fit(x_train, y_train)
            ypred_train = mdl.predict(x_train)
            ypred_test = mdl.predict(x_test)

            # The sum of squares due to regression (SSR) and mean square error (MSE)
            ssr = sst*r2_score(y_train, ypred_train)
            mse = mean_squared_error(y_train, ypred_train)
            # The SSR increase with the new feature added compared to the current model
            if initial_feat_train.empty:
                # Naturally, it is SSR itself if the loop runs for the first time
                ssr_increase = ssr
            else:
                # sorted_results.iloc[0,3] is the previous SSR
                ssr_increase = ssr-sorted_results.iloc[0, 3]

            # F and p values of the model with the newly added feature
            f_value = ssr_increase/mse
            p = 1-scipy.stats.f.cdf(f_value, 1, len(x_train) -
                                    len(x_train.columns)-1)

            # Collect results
            results = [x_train.columns.values, r2_score(y_train, ypred_train), r2_score(
                y_test, ypred_test), ssr, ssr_increase, mse, f_value, p]
            results_features.append(results)

        # Sort results so that the highest R^2 Test value will be on top: The focus of this homework
        # This sorted results frame will be referenced for many times below.
        sorted_results = pd.DataFrame(results_features[1:], columns=results_features[0]).sort_values(
            ["R^2 Test"], ascending=False)
        display(sorted_results)

        # If newly added feature have significant p value, and increases by at least contribution margin R^2 in test set,
        # it has positive impact to the model, accept the added variable.
        if (sorted_results.iloc[0, 7] < p_significance) and (sorted_results.iloc[0, 2] >= initial_R2_test + contribution_margin):

            # Initial features are current model + selected feature and the rest of the features
            # is all the rest. Each of the rest will be subjected to forward selection next time.
            initial_feat_train = pd.DataFrame(
                X_train.loc[:, sorted_results.iloc[0, 0]])
            initial_feat_test = pd.DataFrame(
                X_test.loc[:, sorted_results.iloc[0, 0]])
            rest_feat_train = X_train.drop(
                axis=1, columns=sorted_results.iloc[0, 0])
            rest_feat_test = X_test.drop(axis=1, columns=sorted_results.iloc[0, 0])
            # Also seizing the R^2 values of the current loop, since they are used in comparison of changes
            initial_R2_train = sorted_results.iloc[0, 1]
            initial_R2_test = sorted_results.iloc[0, 2]
            # If R^2 values are in fact better with the new feature, display the results
            print("Currently", sorted_results.iloc[0, 0],
                "is/are the explaining feature(s) because of the increase in the R squared for test.")

        # If R^2 values do not improve with added features, then report, fit the previous model again,
        # summarize the best model and break
        else:
            print("The added features do not improve test R^2 score significantly anymore. The most explaining model has been found.")
            mdl.fit(initial_feat_train, y_train)
            ypred_train = mdl.predict(initial_feat_train)
            ypred_test = mdl.predict(initial_feat_test)
            print("\nThe features that best explain the target:", initial_feat_train.columns.values,
                "\nThe R^2 values for training and test respectively:", r2_score(
                    y_train, ypred_train), r2_score(y_test, ypred_test),
                "\nThe coefficients of the model:", mdl.coef_,
                "\nMean squared error for the training and test respectively:", mean_squared_error(y_train, ypred_train), mean_squared_error(y_test, ypred_test))
            break

        ####Second part eliminates existing ineffective features from the model (Backward elimination)####

        # We will drop each previously added features seperately to see if there are features which
        # do not have significant impact on the model after the last feature has been added. If so, we'll
        # eliminate them from the model.

        # Reset feature effectiveness test variables for they will be filled with the output of the loop.
        feature_test = [["Feature Dropped", "SSR Decrease",
                        "R^2 Decrease Train", "R^2 Decrease Test", "F", "p"]]
        # Reset the ineffective features list
        ineffective_features = []

        # If there are features more than 1, then we can try performing backward elimination
        if len(initial_feat_train.columns) > 1:
            # The loop iterates over the current features one by one
            for i in initial_feat_train.iloc[:, 1:].columns:
                # The train and test sets consist of all added features except the iteration feature
                x_train = initial_feat_train.drop(axis=1, columns=i)
                x_test = initial_feat_test.drop(axis=1, columns=i)
                mdl.fit(x_train, y_train)
                ypred_train = mdl.predict(x_train)
                ypred_test = mdl.predict(x_test)
                # The sum of squares due to regression (SSR): Explained error
                ssr = sst*r2_score(y_train, ypred_train)
                # F value: (SSR of the full model - SSR reduced model) divided by MSE of original model
                ssr_decrease = sorted_results.iloc[0, 3]-ssr
                f_value = ssr_decrease/sorted_results.iloc[0, 5]
                # Calculate p for significance with f(1,n-k-1)
                p = 1-scipy.stats.f.cdf(f_value, 1, len(x_train) -
                                        len(initial_feat_train.columns)-1)

                if p > p_significance:
                    ineffective_features.append(i)

                # Collect results
                results = [i, ssr_decrease, initial_R2_train-r2_score(
                    y_train, ypred_train), initial_R2_test-r2_score(y_test, ypred_test), f_value, p]
                feature_test.append(results)

            # Display the impact of eliminating each existing feature
            sorted_results_ineffective = pd.DataFrame(
                feature_test[1:], columns=feature_test[0]).sort_values("p", ascending=True)
            display(sorted_results_ineffective)

            # If there are at least 1 feature to drop, print and update initiation variables
            if len(ineffective_features) > 0:
                print("\nThe results show that feature(s)", ineffective_features,
                    "proved effective no more. It/they will be dropped from the model.")
                # Set the latest datasets and variables of the model
                initial_feat_train = initial_feat_train.drop(
                    axis=1, columns=ineffective_features)
                initial_feat_test = initial_feat_test.drop(
                    axis=1, columns=ineffective_features)
                rest_feat_train = rest_feat_train.join(
                    X_train.loc[:, ineffective_features])
                rest_feat_test = rest_feat_test.join(
                    X_test.loc[:, ineffective_features])
                # To provide R^2 comparison to next feature selection loop, we will fit the model
                # with the updated train, test sets.
                x_train = initial_feat_train
                x_test = initial_feat_test
                mdl.fit(x_train, y_train)
                ypred_train = mdl.predict(x_train)
                ypred_test = mdl.predict(x_test)
                # Now update the R^2 for sake of comparison in next forward selection loop
                initial_R2_train = r2_score(y_train, ypred_train)
                initial_R2_test = r2_score(y_test, ypred_test)
            else:
                print("There is no ineffective feature.")

        else:
            pass