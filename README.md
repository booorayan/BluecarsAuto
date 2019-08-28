# Hypothesis Testing and Descriptive Analysis of Autolib Dataset

#### The project entails determining the distribution of the total number of bluecars taken from stations in Paris. Knowledge on the distribution of bluecars taken during the week can be used in strategic planning, for instance, increasing the number of cars available during peak days.

#### The dataset used for this project was obtained from here http://bit.ly/DSCoreAutolibDataset and it includes data collected in 2018. 


#### 28/08/2019

## Requirements
The libraries required for this project included:

      pandas - for performing data analysis and cleaning.

      numpy - used for fast matrix operations.

      matplotlib - used to create plots.

      seaborn - used for creating plots.  
      
      sklearn - machine learning library      
  
The language used was python3 and the regression model chosen for the project was a linear regression model.  

## Description
The objective of the project was to determine if there is a difference in the total number of bluecars taken from stations on Paris between Monday and Friday. Using this knowledge, relevant firms can plan accordingly with regard to availability of bluecars. 

Bivariate, univariate and multivariate analysis were conducted on the data to help determine appropriate features for modelling. 

A linear regression model was used to help determine the features that are statistically insignificant and can be excluded from the model. 


### Experiment Design
This project followed the CRISP-DM methodology for the experiment design. The CRISP-DM methodology has the following steps:

####   1.   Problem understanding: 
Entailed gaining an understanding of the research problem and the objectives to be met for the project. External research was conducted to gain an understanding of electric car sharing services. 
The metrics for success were also defined in this phase.
Some metrics for success included:
  *   Get a sample(s) of the data 
  *   Determine the p-value
  *   Reject or accept/fail to reject the null hypothesis
   
####   2.   Data understanding: 
Entailed the initial familiarization with and exploration of the dataset, as well as the evaluation of the quality of the dataset provided for the study.

            # loading the dataset and previewing the first 5 observations 
            url = 'http://bit.ly/DSCoreAutolibDataset'

            autoe = pd.read_csv(url)
            autoe.head()
            
            # reading the columns of the dataframe
            autoe.columns
            
            # checking the datatype of the columns and no. of non-null columns
            autoe.info()
   
####   3.   Data preparation: 
Involved data cleaning/tidying the dataframe to remove missing values and ensure uniformity of data. 
   
            # replacing whitespaces in the columns with underscores and\
            converting column names to lowercase to ensure uniformity
            
            autoel.columns = autoel.columns.str.replace(' ', '_').str.lower()
            
            # checking for sum of duplicate values
            autoel.duplicated().sum()
            
            # checking for the sum of missing values in each column
            autoel.isnull().sum()
            
            # converting date column to datetime
            autoel.date = pd.to_datetime(autoel.date)
 
The EDA revealed no missing values or duplicate data in the dataframe.

####   4.   Modelling: 
Involved the processes of selecting a model technique, selecting the features and labels for the model, generating a test design, building the model and evaluating the performance of the model. 
   
The bluecars_taken_sum column was selected as the target/label (i.e. y) for the model.
      
The features for the model (i.e. X) are all the columns excluding of the postal_code, bluecars_taken_sum, year, and date columns.

           # specifiying out features and target variables 
            features = autoli.drop(['bluecars_taken_sum'], 1)
            target = autoli['bluecars_taken_sum']

The model chosen for this project wasa linear regression model

            # splitting data to training and test sets using train_test_split()
            # 30% of the data wil make up the testing set and the rest will be the training set

            features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=45)

            # creating and training the model by fitting the linear regression model on the training set
            linre = LinearRegression()
            res = linre.fit(features_train, target_train)

            # grabbing predictions off/from the test set 
            pred = linre.predict(features_test)

            # calculating the coefficient of determinantion, R2
            r2_score(target_test,pred)

The R squared value indicated that a linear model explains approx. 94.63% of response data variability
Therefore, a linear model can serve as a good fit.

The OLS Regression summary revealed that the slots_taken_sum column is statistically insignificant and can be excluded from our model.

####   5.   Evaluation: 
Groupby() method was used to challenge the result of hypothesis testing. The population was grouped by dayofweek and the results revealed a slight difference in the total number of bluecars taken between Monday and Friday.

            # grouping the population by day of week and displaying the sum of bluecars taken each day

            group_cars = autob.groupby('dayofweek')[['bluecars_taken_sum']].sum()
            group_cars.head()

### Conclusion


*   Since the determined p-value is above the level of significance of 0.05 we FAIL to reject the null hypothesis.

*   Even though we failed to reject the null hypothesis, it does not implicitly mean that we accept the null hypothesis.

*   Grouping the data by dayofweek reveals that the total number of bluecars taken are relatively similar, differing by slight percentages.

*   An assumption taken is that highly correlated features/variables represent the same information and one used inplace of the other. As a result, the columns bluecars_returned_sum', 'utilib_returned_sum', 'utilib_14_returned_sum were not used/included in the analysis


### License

Copyright (c) 2019 **Booorayan**
