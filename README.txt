Section 1: Project Overview
The Black-Box Optimization (BBO) capstone project is carried out as part of Professional Certificate in Machine Learning and Artificial Intelligence. The project focuses on maximizing eight unknown objective functions with dimensionalities ranging from 2 to 8 variables.
The functions were treated as true black boxes: their analytical forms, gradients, and structural properties were completely unavailable. The only permitted interaction was to submit an input vector and receive the exact scalar output. More specifically, each function is queried only once per week, resulting in a total of 13 evaluations per function over the full duration of the project. So far there are only 3 queries.
The objective of this project was to develop and refine a method capable of modelling unknown functions from a very small number of observations and selecting new evaluation points within the strict evaluation limit.
In real world machine learning, black-box optimization applies when we have controllable inputs and an output we want to maximize, but the relationship between them is unknown, complex, or too difficult to model directly. Because each test is costly, slow, or limited, every new evaluation must be chosen strategically rather than randomly, which is exactly the type of situation black-box optimization is designed to address.
In finance, black-box optimization can be used to tune portfolio allocations, trading strategy parameters, or risk policy thresholds when the relationship between inputs and performance is complex and only observable through back testing or simulation. 
Section 2: Inputs and Outputs
There are 8 unknown functions of varying dimensionality and starting dataset size. Each dataset grows by 1 data point each week, allowing the model to be updated with new information to inform its next decision. The dimensionality of the inputs vary but the output is always a scalar
The functions are summarised below:
Function	Input	Input size (rows)	Output
1	x = [x_1, x_2]	10	y=F(x)
2	x = [x_1, x_2]	10	y=F(x)
3	x = [x_1, x_2,x_3]	15	y=F(x)
4	x = [x_1, x_2,x_3,x_4]	30	y=F(x)
5	x = [x_1, x_2,x_3,x_4]	20	y=F(x)
6	x = [x_1, x_2,x_3,x_4,x_5]	20	y=F(x)
7	x = [x_1, x_2,x_3,x_4,x_5,x_6]	30	y=F(x)
8	x = [x_1, x_2,x_3,x_4,x_5,x_6,x_7,x_8 ]	40	y=F(x)

Queries must use the format x_1-x_2-x_3-…-x_n  , with each variable given to exactly 6 decimal places. Each query returns a single continuous output value that depends on the selected function.
Section 3: Challenge Objectives
The objective of this project is to maximize the scalar output value of a set of eight hidden black-box functions by identifying input locations that approximate their global optima. This task is complicated by a strict evaluation budget and complete uncertainty regarding the structural properties of the behaviour of the function.
The challenge is shaped by three main constraints: only a small number of evaluations is permitted, making brute-force or dense search infeasible; queries are submitted across iterative rounds, so each decision must be made before receiving the next batch of results; and no information is available about the analytical form, gradients, smoothness, noise, or multimodality of the functions.
Section 4: Technical Approach
Gaussian Process surrogate model
Because the true objective functions are unknown, I approximated each one using a Gaussian Process surrogate model. A GP is a probabilistic model that estimates both:
•	the expected value of the function at unseen points
•	the uncertainty associated with those predictions
This makes GPs particularly well suited to black-box optimization. Unlike purely deterministic regression models, they do not only say where the function might be high - they also indicate where the model is uncertain, which is critical when deciding where to sample next.
At each iteration, the GP was updated using all previous observations for a given function. This allowed the model to become progressively more informative as more queries were collected.
Expected Improvement acquisition function
To decide where to query next, I used the Expected Improvement (EI) acquisition function.
EI evaluates candidate points according to how much improvement they are expected to provide over the best function value observed so far. In the first two rounds I have mainly applied exploration and in the last run I have implemented exploitation by adjusting the parameter accordingly.
Sequential optimization process
For each of the 8 functions, the optimization loop followed this general process:
1.	Collect existing observations for the function
2.	Fit or update the Gaussian Process model
3.	Optimize the Expected Improvement acquisition function over the search domain
4.	Select the next query point suggested by the acquisition function
5.	Evaluate the black-box at that point
6.	Augment the dataset with the new observation
7.	Repeat
As the project proceeds, I will be exploring other techniques to enhance the my next candidate selection. This includes comparing the GP and EI approach with random search, grid search, or other Bayesian optimization acquisition functions; testing alternative acquisition strategies such as Upper Confidence Bound (UCB) or Probability of Improvement (PI); experimenting with different GP kernels and hyperparameter settings; and studying how performance changes as dimensionality increases.


