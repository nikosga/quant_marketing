# Objective Function
The objective function is the maximization of profit over the set of ad channels $i \epsilon [1,...,N]$ as a function of the marketing spend $c_i$, with $q_i$ being the acquired customers through channel $i$:

$$max_{c_i} \sum_{i=1}^N \pi_{i} (c_i) = max_{c_i} \sum_{i=1}^N q_i(c_i) * LTV_i - c_i$$

$$s.t. \sum_{i=1}^N c_i <= budget$$

### Acquisitions Demanded

We can represent diminishing returns through linear-log response curves. In that case:

$$q_i(c_i) = b_i*log(c_i)$$


### Setting up the Lagrangian Equation
$$ L = \sum_{i=1}^N b_i*log(c_i) * LTV_i - c_i + \lambda*(budget - \sum_{i=1}^N c_i)$$

# First Order Conditions

For every channel the following condition applies:

$$ \frac{LTV_i*b_i}{c_i}  - 1 - \lambda = 0 \, (1)$$

And the condition for $\lambda$
$$ \sum_{i=1}^N c_i = budget \, (2)$$

For 2 channels:

$$(1) \Rightarrow \frac{LTV_1*b_1}{c_1} = \frac{LTV_2*b_2}{c_2}\Rightarrow $$

$$(2) \Rightarrow c_2 = budget - c_1$$

Putting (2) into (1):

$$\frac{LTV_1*b_1}{c_1} = \frac{LTV_2*b_2}{budget - c_1}\Rightarrow $$

$$ \frac{budget - c_1}{c_1} =  \frac{LTV_2*b_2}{LTV_1*b_1} \Rightarrow$$

$$budget = c_1 [1 + \frac{LTV_2*b_2}{LTV_1*b_1}] \Rightarrow$$

$$c_1 = \frac{budget}{1 + \frac{LTV_2*b_2}{LTV_1*b_1}} , c_2 = budget - c_1$$