# Cryptanalysis of the *Speck 32/64* cipher using *Convolutional Neural Networks*

## Description of the *Speck 32/64* encryption algorithm

### Parameters

Speck is an ARX (Addition, Rotation, XOR) type block cipher. 
* **word size** = 16 bits
* **block** = $(L,R)$ = the encrypted word (which has 32 bits) is divided into two 16-bit subwords
* **key** = 64-bit number
* **rotations** = number of bits rotated to the right or left in the encryption function (default values are $\alpha = 7$ for right rotation on $L$ and $\beta = 2$ for left rotation on $R$)
* **number of rounds** = number of subkeys the initial key is divided into

### Encryption function

Let us define:
* $ROR(L,\alpha)$ right rotation in $L$ by $\alpha$ bits
* $ROL(R,\beta)$ left rotation in $R$ by $\beta$ bits
* $K_i$ subkey of round $i$, derived from the initial key, having an exact length of $w$ bits
* $\oplus$ bitwise $XOR$ operation

Now, for the **encryption** of the message we use:

$$f(L,R) = (L',R')$$

where:

$$L' = ((ROR(L,\alpha) + R) \bmod 2^{w}) \oplus K_i$$

$$R' = ROL(R,\beta) \oplus L'$$

For **decryption** we use the inverse of function $f$:

$$f^{-1}(L',R') = (L,R)$$

where:

$$R = ROR(R' \oplus L', \beta)$$

$$L = ROL(((L' \oplus K_i) - R) \bmod 2^{w}, \alpha)$$

## The Decryption Attack

The attacker observes ciphertexts encrypted in the same way, without knowing the key used for their encryption. The attacker aims to find the last subkey used to encrypt a message. Once the last key is found, the process is repeated until the entire secret key is discovered. All evaluation metrics presented will reflect the algorithms' capacity to return this last subkey.

### Convolutional Neural Network (CNN)

#### Problem Definition
We construct a convolutional neural network (CNN) that acts as a *neural distinguisher*. The network will return a probability $p \in [0, 1]$ to answer the question: "Does the ciphertext pair $(C_1, C_2)$ originate from the encryption of two plaintexts that respect a certain fixed difference?". Complementarily, $1-p$ will represent the probability that the respective pair is formed by completely random bit sequences.

#### Training Data Generation
To train the network, we use plaintext pairs that differ by a fixed XOR value, denoted $\Delta P = (\Delta L, \Delta R)$. The process of generating positive (real) data involves encrypting these pairs:

$$(L,R) \xrightarrow{encrypt} C_1$$

$$(L\oplus \Delta L, R\oplus \Delta R) \xrightarrow{encrypt} C_2$$

Thus, a valid sample for the network will be formed by combining the obtained ciphertext pairs: $(C_1, C_2)$. The complete dataset is obtained by balancing the classes: half consists of positive examples (real encryption pairs with the given difference), and the other half consists of negative examples (where the second ciphertext $C_2$ is replaced with a uniformly generated random value).

#### Network Architecture

The proposed architecture is a residual convolutional network consisting of a residual tower and a prediction head, an architecture that was simplified and improved to reduce the number of parameters without losing accuracy.

**Layer Structure**
* **Input:** A three-dimensional tensor with 3 channels, corresponding to the features extracted from the pair $(C_1, C_2)$. The spatial dimension is 16 bits (according to the Speck32/64 architecture).
* **Residual Block:** The network integrates *depth* successive residual blocks. Each block performs:
    * `Conv1d` (expansion from 3 to 32 channels, kernel = 3, padding = 1).
    * `BatchNorm1d` followed by the non-linear activation function `ReLU`.
    * `Conv1d` (contraction from 32 back to 3 channels, kernel = 3, padding = 1).
    * `BatchNorm1d` followed by `ReLU`.
    * A residual connection that adds the block's original input to its output, to prevent vanishing gradients.
* **Prediction head**: The output of the residual tower, shaped $(3, 16)$, is flattened into a 48-element vector and passed through fully connected layers:
    * `Linear` (48 $\rightarrow$ 64 neurons), `BatchNorm1d`, `ReLU`.
    * `Linear` (64 $\rightarrow$ 64 neurons), `BatchNorm1d`, `ReLU`.
    * `Linear` (64 $\rightarrow$ 1 neuron), followed by the `Sigmoid` function to map the output to the desired binary probability.

#### Training Parameters and Strategy

We will train models for 5, 6 (with *depth* $= 10$), and 7 encryption rounds (with *depth* $= 1$), and the training process uses the following hyperparameter configuration:
* **Loss Function:** Mean Squared Error (`MSELoss`).
* **Optimizer:** `Adam`, with an $L2$ regularization and a weight decay of $10^{-5}$.
* **Learning Rate:** Dynamic scheduler of type `OneCycleLR`, with a maximum rate of $10^{-3}$.
* **Dataset:** $10^7$ total samples ($9 \cdot 10^6$ for training, $10^6$ for validation).
* **Batch size**: 5000 examples per batch.
* **Duration:** Training runs for 200 epochs, retaining the model state with the best accuracy on the validation set.

The total training time for the 3 networks is approximately 64 hours using an *11th gen i7 CPU*, a *GTX 1650 GPU* (with *4GB VRAM*), and *16 GB RAM*. The same system was used for the attack methods presented later.

#### Training Results

The **5-round** model has an accuracy of 92.74%, the **6-round** one has 78.79%, and the **7-round** one 55.14%.
     
If we tried to use the same training strategy for models of 8 or more rounds, the accuracy would be around 50%, thus equivalent to a random class guess, a result that cannot be used by any of the cryptanalysis algorithms presented below.

### Utilizing CNN Probabilities

Once trained, the neural distinguisher (DND) is not used in isolation, but as a central component in the subkey recovery phase (network inference on partially decrypted data). Next, we will present two methods of aggregating probabilities to determine the correct subkey.

#### Sum of Logits

Because the network's accuracy for a single ciphertext pair is limited, ciphertext structures generated based on neutral bits are used. The network's responses for all pairs in the structure are aggregated to formulate a confidence score for each candidate key.

**Notations:**

* $f_0(X) = P(real|X)$: probability returned by the CNN that the input data $X$ originates from a real encryption.
* $X_i(K) = f^{-1}(C_i, K)$: the result of the partial decryption (by one round) of the ciphertext pair $C_i$ using the candidate subkey $K$.
* $p_i(K) = f_0(X_i(K))$: the probability estimated by the network for pair $i$ decrypted with key $K$.
* $l_i(K) = \log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$: transformation of the probability into *log-odds*.

**Description:**

Assume we have $n$ ciphertext pairs $(C_{i1}, C_{i2}), i=\overline{1,n}$, obtained from a structure, and we know the length of the last subkey. For each candidate key $K$ from the corresponding key space, we partially decrypt the $n$ pairs. The CNN evaluates each result, providing a probability $p_i(K)$. The total score for the candidate subkey $K$ is calculated by summing the log-odds values:

$$S(K) = \sum_{i=1}^{n}\log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$$

The maximum value of the score $S(K)$ will indicate the most probable subkey.

**Theoretical foundation of the method**

This approach is optimal under two strict assumptions:

* The CNN is *Bayes-optimal*, meaning the predicted probability perfectly reflects the real distributions:

  $$P(real|X) = \frac{P_{real}(X)}{P_{real}(X) + P_{random}(X)}$$
  
  where $P_{real}(X)$ is the probability density under the hypothesis that the input comes from the cipher's distribution, and $P_{random}(X)$ is the density under the hypothesis of a uniform distribution.

* The $n$ partially decrypted examples $X_i(K)$ are conditionally independent given the key $K$.

Based on the first assumption, we deduce that:

$$\frac{f_0(X)}{1-f_0(X)} = \frac{P_{real}(X)}{P_{random}(X)} \iff \log_2\left(\frac{f_0(X)}{1-f_0(X)}\right) = \log_2\left(\frac{P_{real}(X)}{P_{random}(X)}\right)$$

Therefore, the score formula becomes equivalent to summing the log-likelihoods:

$$\forall i, K: \; l_i(K) = \log_2\left(\frac{p_i(K)}{1-p_i(K)}\right) = \log_2\left(\frac{P_{real}(X_i(K))}{P_{random}(X_i(K))}\right)$$

If the network is Bayes-optimal, maximizing $S(K)$ is a theoretically optimal decision for classifying independent sequences. In practice, because the CNN is only an approximation of the ideal distribution, the method provides only an empirical estimator, but quite a robust one considering the simplicity of this algorithm.

**Algorithm Evaluation**

To attack an $n$-round system, we will use the model trained on $n-1$ rounds. We will consider the attack successful only when the true key is among a top 32 keys considered by the algorithm as the most probable to be the real key. Thus, we have:

* For the 6-round attack, we used the model trained on 5 rounds and the success rate is 100%, and the real key is on average at rank 1.50 in the key ranking given by the algorithm.
* For the 7-round attack, we used the model trained on 6 rounds and the success rate is 100%, and the real key is on average at rank 1.60 in the key ranking given by the algorithm.
* For the 8-round attack, we used the model trained on 7 rounds and the success rate is 100%, and the real key is on average at rank 5.10 in the key ranking given by the algorithm.

#### Bayesian Key Search

When trial decryption is done for a single round, the randomization hypothesis for wrong keys often fails, especially in the case of lightweight ciphers like Speck32/64. To solve this problem and streamline the search, the *Bayesian Key Search* (BKS) algorithm is used, improved by adopting and adapting it to guarantee the retention of optimal keys.

**Profile Precalculation (WKRP)**

A profile of the network's response for wrong keys (WKRP - *Wrong Key Response Profile*) is generated. For various differences between the real key and the test key ($\Delta k = k_i \oplus k$), the partially decrypted ciphertexts are evaluated. Transforming the DND results into log-odds, we obtain the mean $\mu_{\Delta k}$ and standard deviation $\sigma_{\Delta k}$ for each possible difference $\Delta k$.

**Iterative Search Algorithm**

Unlike brute force (evaluating the entire key space), the improved BKS algorithm executes $\ell$ iterations to successively refine a restricted set of candidates. The process consists of the following steps:

1. It starts with a set $S$ consisting of $n_{cand}$ candidate keys. To prevent losing the real key due to statistical fluctuations, if there is a global optimal key determined in previous steps or batches ($K_{best}$), it is forcibly retained in the current candidate set.
2. For each candidate key $k_i \in S$ and each ciphertext pair $j$ from the structure, we decrypt one round, pass the result through the DND to obtain the probability $v_{j,k_i}$, and then calculate the log-odds:
   
   $$z_{j,k_i} = \log_2\left(\frac{v_{j,k_i}}{1-v_{j,k_i}}\right)$$

3. Calculate the mean log-odds score for each candidate key $k_i$:
   
   $$m_{k_i} = \frac{1}{n_{cts}}\sum_{j=0}^{n_{cts}-1}z_{j,k_i}$$

4. Iterating through the entire space of possible keys $k \in \mathcal{K}$, a penalty score $\lambda(k)$ is calculated, representing the weighted Euclidean distance:
   
   $$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

5. The set $S$ is updated by retaining the $n_{cand}$ keys $k$ that minimize the score $\lambda(k)$ and moves to the next iteration.

**Theoretical foundation of the method**

The efficiency of the BKS method relies on the following assumptions:
* The empirical log-odds means ($m_{k_i}$) follow a normal distribution dictated by the key difference.
* The precalculated parameters $\mu$ and $\sigma$ in the WKRP table accurately reflect the real distribution.

Assuming the mean of the log-odds obtained with key $k_i$ is normally distributed relative to the real key profile $k$:

$$m_{k_i} \sim \mathcal{N}(\mu_{k_i\oplus k}, \sigma_{k_i\oplus k}^2)$$

The likelihood function for the observed means vector $m$, conditioned on the correct key $k$, becomes:

$$P(m|k) = \prod_{i=0}^{n_{cand}-1}\frac{1}{\sqrt{2\pi\sigma_{k_i\oplus k}^2}}e^{-\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{2\sigma_{k_i\oplus k}^2}}$$

Applying Bayes' theorem (with a uniform prior distribution over the key space $P(k)$) we obtain $P(k|m) \approx P(m|k)$. Transitioning to the logarithmic domain to avoid numerical instability and ignoring constant terms, maximizing the probability $\log_2 P(k|m)$ becomes equivalent to minimizing our error metric $\lambda(k)$:

$$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

**Algorithm Evaluation**

We used the same evaluation method as for *Sum of logits* and obtained the following results:

* For the 6 and 7-round attacks, we obtained a 100% success rate, and the true key is always first in the top keys returned by the algorithm.
* For the 8-round attack, the success rate is 20%, and the real key is, on average, at rank 2.50.

### Fine-Tuning the Model (with hard negative examples)

Although the neural network described earlier achieves high accuracy on the standard dataset, its performance can decrease during the attack phase (at key recovery). In standard data generation, negative examples are created by replacing a valid ciphertext with completely random data. However, in practice, during the attack, the network evaluates ciphertexts that have been decrypted with a wrong candidate subkey. These erroneous decryptions do not produce perfectly random noise but retain certain structural correlations specific to the cipher, a phenomenon that weakens the model's distinction capabilities. To solve this problem, we will fine-tune the model using hard negative examples. 

**Generating hard negative examples**

When generating these examples, we will simulate exactly the scenario encountered during the key search, following these steps:

* Generate and encrypt text pairs for $R$ rounds using the correct keys.
* For a negative sample, the obtained pair is encrypted for one more round using the correct subkey. Immediately after, the pair is decrypted for one round, but this time using a random subkey.

Thus, we obtain a dataset that better mimics the scenarios the model will encounter during inference.

#### Fine-Tuning Results

We trained the models for another 20 epochs using such negative examples, thus increasing their accuracy for pairs of this type by approximately 1%.

### Proposed Models and Hybrid Algorithms
To maximize both execution speed and decryption attack success rate, we implemented and evaluated a series of derived strategies. These range from the direct use of the fine-tuned model to hybrid algorithms. The hybrid algorithms combine the pre-filtering advantages of the *Sum of Logits* (SoL) method with the refined precision of the *Bayesian Key Search* (BKS), simultaneously using the two types of neural networks: the base model (standard trained) and the fine-tuned model (denoted FT).

These strategies can be classified as follows:

* **Fine-Tuned BKS (FT BKS):**
  This approach represents a direct improvement of the base algorithm. It executes the Bayesian Key Search algorithm over the entire key space ($2^{16} = 65536$ possibilities) but uses exclusively the FT model and its corresponding WKRP profile, replacing the base model.

* **SoL $\rightarrow$ Original BKS:**
  Evaluating the entire key space directly using BKS is a computationally expensive operation. Through this hybrid method, we initially use the SoL technique on the base model to approximate very quickly and extract only a narrow top of candidates (64 keys). The resulting set becomes the exclusive search space for the original BKS algorithm, massively reducing execution time while limiting the number of erroneous candidates that could have caused false-positive results.

* **SoL $\rightarrow$ Fine-Tuned BKS:**
  This technique respects the same principle of restricting the search space. After identifying the 64 most probable keys using SoL and the base model, the final refinement is done by applying BKS exclusively with the FT model. The method combines the generalization capability of the base model with the superior accuracy of the FT model when facing hard negative examples.

* **Ensemble BKS:**
  Within this BKS-type implementation, instead of inference on a single model, the neural distinguisher acts as an ensemble formed by both models (the standard one and the FT one). Their responses are weighted based on the inverse squares of the loss function obtained in the validation process:
  
  $$w_i = \frac{1}{\text{loss}_i^2}, \quad i \in \{1, 2\}$$
  
  The normalized weights thus become:
  
  $$\alpha_1 = \frac{w_1}{w_1 + w_2}, \quad \alpha_2 = \frac{w_2}{w_1 + w_2}$$
  
  The final aggregated probability ($p_{final}$), which will later be transformed into log-odds for the Bayesian algorithm, results from the weighted sum of the individual probabilities predicted by the two models:
  
  $$p_{final} = \alpha_1 \cdot p_{base} + \alpha_2 \cdot p_{FT}$$

* **SoL $\rightarrow$ Ensemble BKS:**
  The most complex hybrid architecture tests all 65536 keys using SoL and the base model, and the 64 keys that survive this pre-filtering stage are passed on to the *Ensemble BKS* search model to determine the global optimum.

### Optimizing the BKS Algorithm

Although the BKS algorithm brings major improvements in the key recovery success rate, the initial calculation of the WKRP (*Wrong Key Response Profile*) is a computationally highly expensive operation. To build this profile of distributions, the neural network must evaluate millions of ciphertext pairs for absolutely all $2^{16} = 65536$ possible key differences ($\Delta k$). Because this profile remains identical for any attack made by the same model on the same number of rounds, we implemented an offline caching mechanism. Instead of recalculating the profile for each new instance or attack scenario, the WKRP tables (vectors of means $\mu$ and standard deviations $\sigma$) are generated only once for each trained model (both the base and the fine-tuned one) and are serialized to disk as files. In the actual attack phase, the profile is simply loaded into VRAM, an operation much faster than calculating the profile, and also faster than the *sum of logits* algorithm for which such precalculation is impossible.

Thus, the BKS algorithm runs the neural network inference exclusively for a very small active set of candidate keys ($n_{cand} = 64$). To explore and evaluate the rest of the 65536 key space, BKS **no longer** calls the neural network. Instead, it uses the $\mu$ and $\sigma$ values from the WKRP cache to calculate the weighted Euclidean distance $\lambda(k)$:

$$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

This Bayesian evaluation is reduced to elementary vectorized mathematical operations on tensors, which are executed almost instantly on the massively parallel architecture of a GPU, completely bypassing the need for new inferences with the CNN.

## Evaluation and Results of Cryptanalysis Methods

To compare the efficiency of the proposed methods, the evaluation was carried out through an automated testing framework that simulates attack scenarios on 6, 7, and 8 rounds. 

### Evaluation Methodology
The evaluation process works through a continuous loop, where each iteration represents a new challenge. For each challenge, the following parameters are generated:
* A randomly generated target subkey, representing the attack objective to be recovered.
* A specific number of plaintext and ciphertext structures: 32 structures for the 6-round attack, 64 structures for 7 rounds, and 128 structures for 8 rounds.

In each iteration, the ciphertexts are sequentially passed to six distinct key recovery methods:
1. **M1 (Original BKS):** The base method, using the BKS algorithm with the standard trained model.
2. **M2 (Fine-Tuned BKS):** The BKS algorithm evaluated exclusively with the fine-tuned model.
3. **M3 (SoL $\rightarrow$ Original BKS):** Reducing the search space to 64 candidates using the *Sum of Logits* (SoL) method and evaluating them with the original BKS.
4. **M4 (SoL $\rightarrow$ Fine-Tuned BKS):** Search space reduction through SoL, followed by BKS evaluation using the fine-tuned model.
5. **M5 (Ensemble BKS):** The BKS algorithm using an ensemble formed by both models (standard and fine-tuned).
6. **M6 (SoL $\rightarrow$ Ensemble BKS):** Space reduced through SoL to the top 64 keys, subsequently evaluated with the Ensemble BKS method.

### Recorded Metrics and Stopping Criterion

For each method and each challenge, the algorithm measures the execution time and verifies if the predicted subkey perfectly matches the real target subkey. Based on this data, the overall accuracy (number of correct predictions divided by the total number of runs) and the average execution time are dynamically calculated and updated. Unlike previous evaluations, we will consider the attack successful only if the correct key is the first in the top keys predicted by the algorithm. Thus, the accuracy will be considerably lower than in the previous analysis.

To ensure a statistically relevant evaluation, the testing environment requires a minimum of 10 runs before verifying any dominance condition. The infinite evaluation loop stops only when at least one of the proposed methods (M2 - M6) demonstrates a clear dominance over the base method (M1). This dominance is defined by meeting one of the following two conditions:
* **Superior accuracy:** The accuracy of the proposed method is strictly higher than the accuracy of the base method (M1).
* **Speed dominance:** The proposed method achieves an accuracy greater than or equal to that of method M1 (both having an accuracy strictly greater than 0), but records a strictly lower average execution time.

The moment either of these success criteria is met, the scenario run stops, the environment displays the final summary with the times and accuracies of all 6 methods, and the entire history of the associated datasets (up to a fixed limit of 1000 challenges) is automatically saved to disk.

### Attack Results

To validate the performance of the proposed methods, the testing loop was run for the three distinct attack scenarios (6, 7, and 8 rounds). In all three cases, the hybrid algorithms and those based on the fine-tuned model demonstrated clear superiority over the original reference method (M1 - Orig BKS), with the results maintaining a consistent pattern of dominance. Since this evaluation method is constructive, we also found a dataset where at least one of the proposed algorithms is more performant than the reference algorithm.

#### Accuracy Analysis
In the first two attack scenarios (6 and 7 rounds), methods M2 (FT BKS), M4 (SoL $\rightarrow$ FT BKS), and M5 (Ensemble BKS) quickly reached a 100% accuracy, surpassing the 90% performance of the base algorithm (M1). Moreover, in the third scenario (8 rounds), considered an extreme case due to probability degradation and massive noise in the data, the pure Bayesian methods (M1, M2, M5) recorded a 0% success rate. In direct contrast, methods using *Sum of Logits* for pre-filtering (M3, M4, M6) managed to recover the correct key in 20% of cases, demonstrating the vastly superior robustness of hybrid architectures.

#### Execution Time Analysis
The massive differences in execution time confirm the theoretical advantage of the offline caching mechanism explained earlier. Direct BKS methods (M1 and M2) are extremely fast, completing a challenge in under a second (averaging 0.54s - 0.78s). Method M5 (*Ensemble BKS*) requires approximately double this time (1.06s - 1.52s), a logical and efficient increase considering it simultaneously processes the responses of two different neural networks. 

On the other hand, methods integrating the SoL component (M3, M4, M6) are constrained by the necessity to perform CNN inference across the entire 65536 key space. This aspect leads to significantly higher execution times, ranging between 155 and 342 seconds per challenge, perfectly illustrating the speed-complexity tradeoff, albeit with higher performance in some cases.

Secondly, optimizing through offline caching of the WKRP profile eliminates the need for repeated inference, transforming the *Bayesian Key Search* (BKS) algorithm into a much faster process than the classic *Sum of Logits* (SoL) approach.

The main contribution is represented by the proposed hybrid algorithms (M2 - M6), which offer an ideal tradeoff between speed and accuracy. For 6 and 7-round attacks, the *FT BKS* and *Ensemble BKS* methods achieve 100% accuracy in a maximum time of 2 seconds. For the 8-round scenario, where classical methods fail completely (0% success) on the dataset found, the use of SoL pre-filtering combined with *Ensemble* analysis represented the only viable solution, recovering the key in 20% of cases.
