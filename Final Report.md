# Machine Learning Techniques in Suicide Rate Prediction
![1_sldzYWPPlz08SGK61VRpqA](https://user-images.githubusercontent.com/94312082/190874958-e7062bd6-c402-48ad-a7d5-e83d995dd169.jpeg)

Link to youtube video - https://www.youtube.com/watch?v=hXhDTA4ZKCE

Link to PPT - https://github.com/sAKHILAREDDY/DATA606_Akhila/blob/main/Suicide%20Rate%20Prediction.pptx

# INTODUCTION
Suicides have become increasingly concerning and have received a lot of attention in today's society. Suicides are regarded to be most frequently caused by depression. Other factors to consider include economic factors, societal factors and incurable diseases. Chatbots based on AI have been developed to prevent people from committing suicide, although their accuracy is only about 75%. To reduce the number of suicide attempts in the future, we must employ machine learning algorithms to accurately anticipate suicide attempts. Preliminary data analysis would reveal suicide numbers as well as the relationship between various factors and the amount to which they contribute. To better comprehend the trends in suicide attempts, a graphical representation would be presented. Several Python libraries will be used, and multiple models will be constructed to see which model has the least error. This research investigates how suicide rates and the factors that influence them can be predicted using machine learning algorithms. 
# IMPORTANCE OF CURRENT STUDY
* The World Health Organization estimates that 800,000 people die by suicide each year, making it the 18th biggest cause of death. Suicide was the tenth largest cause of death in the United States in 2018, resulting to a decrease in average life expectancy in the United States. Suicide is a global phenomenon that affects people of all ages. According to some estimates, for every adult who died by suicide, more than 20 others attempted suicide. Suicide is a worldwide problem; in 2016, 79 percent of suicides took place in low- and middle-income nations.
* People today suffer from serious physical and psychological diseases because of a range of internal and external circumstances. Although depression is more common in people in their 30s and 40s, it can also be found in children and the elderly due to academic stress and interpersonal relationships. Because people with mental illnesses are stigmatized in society, they frequently conceal their sickness. Self-harm and suicide attempts are also influenced by economic conditions and drug and alcohol usage.
* The first step in suicide prevention can be thought of as a categorization task aimed at precisely identifying persons who are at risk of suicide within a given time frame, allowing for preventive action. However, the greatest meta-analysis of suicide prediction looked at 365 researches and found that predictions based on individual risk or protective factors had poor predictive accuracy and have improved little over time.
# EXPECTATIONS FROM CURRENT STUDY
* Data analysis is performed in order to classify data so that a set of preventive measures can be implemented in the future. This can provide information on the causes of suicide in a specific state and year. The dataset can also show whether the suicide rate for a specific cause has increased or decreased. Not only will analysis and classification give preventative strategies, but they will also enable comparisons to determine whether the suicide incidence has increased or decreased over time. The findings of this study could be applied to a variety of approaches to solve the problem. A thorough investigation was conducted into global suicide information. The following chapters are methodically designed to provide the reader with an organized interpretation.
* Early identification of people who are at risk of suicide is critical for suicide prevention. Machine learning is emerging as a possible strategy for achieving this goal. The aim of our research is to develop a machine learning model for predicting suicide attempts. Some of the models employed are K Nearest Neighbor, Decision Trees, and Random Forest Regression. The effectiveness of these algorithms is compared in this study.
# OUTCOMES INTENDED TO ACHIEVE  
* This analysis is aimed at explaining how different machine learning algorithms can be used in predicting suicide rates based on relevant factors collected in the dataset.
* The analysis carried out will also provide knowledge about areas of improvement to the government and other organizations working towards suicide prevention and counselling so that effective steps can be taken.
# METHODS AND TECHNIQUES
## Data Acquisition
The Suicide Rates Overview 1985 to 2016 dataset which is taken from Kaggle consists of 27820 rows and 12 columns. It is compiled from four different datasets (United Nations Development Program (HDI), World Bank, World Health Organization, and Szmali) to identify any attributes that correlated with suicide rates globally. The data is readily available and downloadable in CSV format.
## Cleaning and Normalization
After acquiring the dataset, we will remove null values or redundant rows, drop repeated columns, and perform outlier analysis and treatment. Country, year, sex, age and generation are all non-numerical labeled columns that will be transformed to numerical labels using SkLearn's LabelEncoder. Many machine learning estimators need dataset standardization: if the individual features do not resemble standard normally distributed data, they may perform poorly. SkLearn's RobustScaler is used to normalize the numerical columns population, gdp_for_year, and gdp_per_capita.
## Exploratory Data Analysis
In this stage, many data mining techniques are utilized to uncover hidden trends in the dataset and determine the correlation between variables, as well as plot multiple graphs to uncover trends in suicide rates and identify the many causes that contribute to suicide. Several python libraries such as NumPy, seaborn, matplotlib etc. have been used. Various plots have been shown to better visualize the data. Correlations are shown with the help of heatmaps and scatterplots.
## Machine Learning Models
Now we will train multiple machine learning models on our dataset and utilize validation techniques to check for overall fit. Finally, we will present the best model for suicide prediction. K Nearest Neighbor, Decision Tree Regression, Random Forest Regression, XGBoost Regression and Multilayer Perceptrons are implemented and evaluated based on accuracies and RMSE scores.
# IMPLEMENTATION
## Dataset
The dataset has been taken from Kaggle. It has 27,820 rows with 12 columns. Some of the columns are numerical types which include GDP per capita, HDI for year, suicides_no, while others like country, age, sex, generation etc. are categorical. It includes data from over 100 countries from 1985 to 2016.
## Evaluation Metrics
The Evaluation Metrics used are accuracy and RMSE scores. Accuracy is a common evaluation metric for classification problems. It is the number of correct predictions made as a ratio of all predictions made. RMSE is one of the most widely used measures for assessing the precision of continuous data. Because RMSE gives large errors a higher weight than MAE, it should be more useful when large errors are undesirable.
Since XGBoost Regression has the highest accuracy and lowest RMSE, it can be considered the best model.
## Model Training and Evaluation 
Our Machine Learning algorithms are K Nearest Neighbor, Decision Tree, Random Forest, XGBoost, and Multilayer Perceptrons (Deep Learning). Below are the accuracies and RMSE's of each model.

K Nearest Neighbor Regression : Accuracy - 0.771, RMSE- 0.279

Decision Tree Regression : Accuracy - 0.967, RMSE- 0.105

Random Forest Regression - Accuracy - 0.988, RMSE- 0.063

XG Boost Regression - Accuracy - 0.997, RMSE- 0.029

Multilayer Perceptrons : Deep Learning - Accuracy - 0.887, RMSE- 0.195

# RESULTS DISCUSSION

Data analysis helped us understand several underlying trends in suicide attempts over the years 1985 and 2016. Coming to the performance of the four machine learning models - Among all the trained models, XGBoost has the highest accuracy and lowest RMSE. This is because XGBoost is very good in execution Speed & model performance. Random forest had an accuracy of 98.9%, followed by Decision Tree, Multilayer Perceptrons and K-Nearest Neighbors with 96.7, 88.7 and 77.1 % respectively.

# CONCLUSION 

This analysis was aimed at explaining how different machine learning algorithms can be used in predicting suicide rates based on relevant factors collected in the dataset. Although there have been several successful high precision models, there is still potential for development. Preliminary data analysis showed some surprising findings which includes teen men are more likely to commit suicide. Machine learning algorithms like XGBoost and Random Forest Regression consistently outperformed other algorithms and had the highest accuracy and precision. The analysis carried out will also provide knowledge about areas of improvement to the government and other organizations working towards suicide prevention and counselling so that effective steps can be taken.

# FUTURE WORK

* This project can be further improvised by combining multiple data sets related to suicides and performing in-depth analysis.
* Some statistical tests- hypothesis testing can be performed which can extract valuable insights.
* Sentiment Analysis can be used to figure out in which social media people feel more free to talk about their mental health.

# REFERENCES

Boudreaux, E. D., Rundensteiner, E., Liu, F., Wang, B., Larkin, C., Agu, E., Ghosh, S., Semeter, J., Simon, G., & Davis-Martin, R. E. (2021). Applying Machine Learning Approaches to Suicide Prediction Using Healthcare Data: Overview and Future Directions. Frontiers in psychiatry, 12, 707916. https://doi.org/10.3389/fpsyt.2021.707916

Gen-Min Lin, Masanori Nagamine, Szu-Nian Yang, Yueh-Ming Tai, Chin Lin, Hiroshi Sato, “Machine Learning Based Suicide Ideation Prediction for Military Personnel”, IEEE Journal of Biomedical and Health Informatics, vol. 24, issue: 7, July 2020

Hasmitha Bhutham, “Suicide rates analysis and prediction” , December 2020.

Mrs. B. Ida Seraphim , Subroto Das , Apoorv Ranjan, 2021, A Machine Learning Approach to Analyze and Predict Suicide Attempts, INTERNATIONAL JOURNAL OF ENGINEERING RESEARCH & TECHNOLOGY (IJERT) Volume 10, Issue 04 (April 2021).








