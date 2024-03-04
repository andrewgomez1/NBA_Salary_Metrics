# NBA_Salary_Metrics
Predicting NBA players' salaries based on in-game statistics

> **This Project**


This project utilized advanced statistical techniques in R to examine the relationship between NBA player statistics and their salaries. Through correlation tests, ridge regression analysis, and the development of a predictive model, the study aimed to identify which player performance metrics have the most significant impact on salary determination in professional basketball, providing valuable insights for player evaluation and contract negotiations.

> **Data Wrangling**


- Narrowed data to just 2015-2016 season stats
- Merged players_stats and salaries_data to a new data frame called "stalaries"
- Removed rows that contained players that played under 20 games
- Split data into training set and testing set for predictive analysis

> **My Methods/Findings**


Correlation Test: I created a correlation plot to visualize the individual correlations between all variables and determine which had the strongest with AdjustedSalary.

Multiple Linear Regression: I ran multiple linear regression and compared p-values. The p-values for free throws made and minutes per game were greater than 0.1, thus ruling these two variables out while the other five variables were able to create a regression model with an adjusted R-squared = 0.449.

Test for Multicollinearity: I tested for multicollinearity by calculating the variance inflation factor. A VIF > 5 indicates that there is moderate correlation between variables and possible multicollinearity among the data. 

Ridge Regression: I ran a ridge regression to find the most optimal coefficients for my linear model that will yield the lowest RMSE. Unfortunately, the root mean square error was minimized when there were no changes.


![Image](https://github.com/users/andrewgomez1/projects/1/assets/124718350/9750c602-caf8-444e-9089-8e016bff09e9)

> **My Conclusions/Visualizations**


The top five player stats with the strongest correlation to their salary are points per game, rebounds per game, assists per game, field goals made, and turnovers per game. 


![Image](https://github.com/users/andrewgomez1/projects/1/assets/124718350/863bb871-a29c-466f-b3ff-e00ce1e0c49d)

> **Possible Errors**


Some confounding variables to account for are contract length, injuries, off-court successes, and team success.
