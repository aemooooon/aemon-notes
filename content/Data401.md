---
title: 
draft: false
tags: 
date: 2024-06-17
---
# Plots
- Stacked bar-chart: Relationship between two categorical variables.
- Box-plot: One quantitative variable.
- Histogram: One quantitative variable.
- Bar-chart: One categorical variable.
- Side-by-side box-plots:   Relationship between one categorical variable and one quantitative variable.
# Describe sample and population
- The sample is the 100 penguins the data was collected from.
- A reasonable population is all penguins of the subspecies being studied
# Experiment or an observational? Explain it. 

This is an observational study as the researchers are not actively controlling the value of the explanatory variable (walking habits). Instead they are observing peoples behaviours.
  
# Causation? Explain in one or two sentences if implying causation is appropriate, based on the study described. 

As it is an observational study the news website should not use a headline that implies causation.

# Bootstrap samples have to be the same size as the original
# Z-score
- calculation
# Linear regression
- Locate positive negative  residual
- explain appropriate:   Yes, as the observations follow a straight line approximately and there is no ‘fanning’ of residuals (meaning the residuals have approximately constant variance).
- estimate   correlation value
- describe the direction and strength of the relationship
	- A correlation of 0.4 indicates a moderately weak positive relationship
	- more todo.
- interpret slop
	-   When the critic rating of a movie increases by 1 the audience rating is predicted to increase by 0.358
- interpret interception
	- When a critic rates the movie with a zero we predict the audience will rate it at 44.77. This is a meaningful interpretation because a critic rating of 0 and an audience rating of 44.77 are within the range of observable values.
- $R^2$
	- 16.7% of the variability in the Audience Rating is explained by the model.

_Cardinality of a variable = number of unique values of a variable_
In statistics random is not the same as haphazard. Simple random sample（SRS）
Association： Two variables are associated if values of one variable tends to be related to the values of the other variable.
Causation： Two variables are causally associated if changing the value of one variable influences the value of the other variable.
Confounding Variable: A third variable that is associated with both the explanatory and the response variable is called the confounding variable.
A confounding variable can offer a plausible explanation for an association between the explanatory and response variable.
- If the explanatory variable can be randomly assigned, it is not associated with any confounding variables.
- Confounding variables are eliminated if the experiment is well designed.
Randomised Experiment
Control Group (treatment) Placebo and Blinding(double-blinded)
Right Skew(positive)[mode, median, mean]
Correlation(sample(γ)population(ρ)): is a measure of the direction and strength of a linear association between two quantitative variables.

- $H_0$: population mean for x $\mu = 7$
- $H_1$: population mean for y $\mu \lt7 \ or \gt 7 \ or \neq 7$ 
- A p-value is the probability/chance of observing a sample statistic as extreme or more extreme as what has been observed if the Null hypothesis is true
- T-test
	- one sample t-test: "mean mercury content in fish is less than 1.0 parts per million"
	- two sample(independent sample t-test): "average time spent on commute in city A and city B is the same" note: the variances of two groups are equal/not equal formula different. 自变量只能是二分类变量，因变量是2个独立的连续变量(>2用ANOVA)
	- paired t-test: "the phone price in NZ is higher than it is in Aus"
- Chi-square test for association. This is because both variables measured are categorical variables, and researchers wish to determine if there is an association. 2个分类变量
	- $H_0$: There is no association between genotype and type of athelete 
	- $H_a$: There is an association between genotype and type of athelete
- Mann-Whitney test is used to compare two independent groups of ordinal data.(>2用Kruskal-Wallis test)
- ANOVA is used to determine whether the means of three or more groups of data are significantly different from each other
	- one-way: when we want to know how one independent categorical variable affects at dependent variable
	- two-way: ~ how two independent variables, in combination, affect a dependent variable
	- Null hypothesis is that the means of all of the groups are the same
	- The result of two-way ANOVA comparing across each individual categorical variable may produce two p values that are smaller than alpha but the comparison across both variables at the same time may produce a significant higher p value

- Conclusion: 
	- Since the p-value 0.352 is significantly larger than alpha = 0.05, therefore the Null hypothesis should not be rejected and conclude that there is insufficient evidence to show the proportion of students living in university halls is more than 25%
	- Reject the Null hypothesis because 0.048 is less than 0.05. There is sufficient evidence that the mean age children start walking is different to 12 months.
	- No, the same decision would be reached because the p-value is larger than 0.01 so the null hypothesis would not have been rejected.
- Type error: 
	- A type I error is when the true null hypothesis is rejected. In this case it would mean we conclude that the mean age at which children start walking is different to 12 months when in fact it is not.
	- A type II error is failing to reject a false null hypothesis. In this situation it would mean that we did not reject the null hypothesis that the proportion of businesses that were advertised to that are considering becoming a customer of the bank is equal to 20% when in fact the proportion was less than 20%.
- DoF:
	- As this is a test for a difference in two population means the degrees of freedom was calculated by subtracting 1 from the smaller sample size.=
- I am 95% confident that the difference (Group 1 – Group 2) in mean weight gain of rats exposed to excessive noise at night and rats in quiet night environment is between n and n.
