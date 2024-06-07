---
title: 
draft: false
tags:
  - model
  - classification
---
>[!Logistic Regression]
>Logistic regression is a type of _classification_ model that works similarly to linear regression. The difference between this and linear regression is the shape of the curve. While simple linear regression fits a straight line to data, logistic regression models fit an s-shaped curve: Logistic regression is better for estimating Boolean outcomes than linear regression because the logistic curve always produces a value between 0 (false) and 1 (true). Anything between these two values can be thought of as a probability.

![[lr01.png]]
As logistic regression gives us these probabilities, rather than simple true/false values, we need to take extra steps to convert the result to a category. The simplest way to do this conversion is to apply a threshold. For example, in the following graph, our threshold is set to 0.5. This threshold means that any y-value below 0.5 is converted to false—left bottom box—and any value above 0.5 is converted to true—right top box.

Looking at the graph, we can see that when the feature is below 5, the probability is less than 0.5 and is converted into false. Feature values that are above 5 give probabilities over 0.5 and are converted to true.

Notably, logistic regression doesn't have to be limited to a true/false outcome – it can also be used where there are three or more potential outcomes, such as `rain`, `snow`, or `sun`. This type of outcome requires a slightly more complex setup called **[[Multinomial Logistic Regression]].**

![[lr02.png]]
