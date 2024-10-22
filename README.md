# quant_marketing
---

# Objective Function
The objective function is the maximization of profit over the set of ad channels $i \epsilon [1,...,N]$ as a function of the marketing spend $c_i$, with $q_i$ being the acquired customers through channel $i$:

$$max_{c_i} \sum_i \pi_{i} (c_i) = max_{c_i} \sum_i q_i(c_i) * LTV_i - c_i$$

### Acquisitions Demanded

We can represent diminishing returns through linear-log response curves. In that case:

$$q_i(c_i) = b_i*log(c_i)$$

$$\frac{  \partial  q_i (c_i) }{ \partial c_i } = \frac{b_i}{c_i}$$

### Putting it together
$$ max_{c_i} \sum_i b_i*log(c_i) * LTV_i - c_i $$

# First Order Conditions

$$ \partial \frac{ \sum_i b_i*log(c_i) * LTV_i - c_i}{\partial c_i} = 0 \Rightarrow$$

$$ \sum_i (\frac{LTV_i*b_i}{c_i}  - 1) = 0 \Rightarrow$$

For this to happen, all components of the sum should be zero at the same time.

$$ \frac{LTV_i*b_i}{c_i} = 1 \Rightarrow $$

$$ c_i = LTV_i*b_i$$