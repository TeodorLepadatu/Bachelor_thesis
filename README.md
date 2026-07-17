# University of Bucharest
## Faculty of Mathematics and Computer Science
### Computer Science Specialization

**Bachelor Thesis:** Cryptanalysis of the Speck32/64 cipher using convolutional neural networks  
**Graduate:** Lepădatu Teodor  
**Scientific Coordinator:** Lect. dr. Bucataru Mihai  
**Bucharest, July 2026**

---

## 1. Introduction

The accelerated development of resource-constrained systems has led to the emergence of a class of cryptographic algorithms specifically designed to operate efficiently under strict memory, power consumption, and computational conditions. In this context, *lightweight* cryptography has become an important research area, providing solutions adapted to applications such as sensor networks, RFID tags, smart cards, or various integrated devices.

A representative example from this category is the *Speck* cipher, developed to offer a compromise between efficiency and security. The *Speck 32/64* variant, analyzed in this paper, operates on 32-bit blocks and uses a 64-bit key, being specifically targeted at hardware-constrained platforms.

In parallel with the development of cryptographic algorithms, machine learning techniques have begun to be explored in cryptanalysis. These methods allow for the identification of subtle statistical regularities in cryptographic data and can complement classical analysis approaches.

This paper investigates the use of convolutional neural networks in the cryptanalysis of the Speck 32/64 cipher, detailed in Section 2. Section 3 analyzes several training and evaluation strategies for the models, as well as a series of optimizations intended to improve the efficiency of the key search process. The results obtained and presented in Section 4 highlight both the potential of deep learning techniques in cryptanalysis and their potential limitations in scenarios with a high number of encryption rounds.

## 2. Description of the Speck 32/64 Encryption Algorithm

### 2.1 Parameters

**Speck 32/64 Parameters:**
Speck is an ARX (Addition, Rotation, XOR) block cipher with the following parameters:
* **word size** = 16 bits
* **block** = $(L,R)$ = the encrypted word (which is 32 bits) is split into two 16-bit subwords
* **key** = 64-bit number
* **rotations** = the number of bits rotated to the right or left in the encryption function (the used values are $\alpha = 7$ for the right rotation on $L$ and $\beta = 2$ for the left rotation on $R$)
* **number of rounds** = the number of subkeys into which the initial key is divided

### 2.2 Encryption Function

**Elementary Operations:**
* $ROR(L,\alpha)$: right rotation in $L$ by $\alpha$ bits
* $ROL(R,\beta)$: left rotation in $R$ by $\beta$ bits
* $K$: the current round's subkey, derived from the initial key, with an exact length of $w$ bits
* $\oplus$: bitwise XOR operation
* $\equiv_{2^w}$: congruence modulo $2^w$

For the **encryption** of the message, we use:

$$f_K(L,R) = (L',R')$$

where:
$$L' = ((ROR(L,\alpha) + R) \bmod 2^{w}) \oplus K$$
$$R' = ROL(R,\beta) \oplus L'$$

**Decryption Function:**
For **decryption**, we use the inverse of the $f_K$ function:

$$f_K^{-1}(L',R') = (L,R)$$

where:
$$R = ROR(R' \oplus L', \beta)$$
$$L = ROL(((L' \oplus K) - R) \bmod 2^{w}, \alpha)$$

**Proof:**
We demonstrate that decryption using this function indeed restores the initial block.
Consider a word of length $w$ in bits, rotated by $\alpha$ bits to the right and $\beta$ bits to the left. Let $K$ be the subkey of the respective encryption round. Let $f_K$ be the current round encryption function mapping $(L,R)$ to $(L', R')$ with:

$L'=((ROR(L,\alpha)+R)\bmod 2^{w})\oplus K$
$R'=ROL(R,\beta)\oplus L'$

Consider the mapping $g_K$:

$\tilde{R}=ROR(R'\oplus L', \beta)$
$\tilde{L}=ROL(((L' \oplus K)-\tilde{R})\bmod 2^{w},\alpha)$

We want to prove that $g_K$ is the inverse of $f_K \iff g_K \circ f_K = id$ and $f_K \circ g_K = id$.

*Proving $g_K(f_K(L,R))=(L,R)$:*
Let $(L',R')=f_K(L,R)$. Consider $S:=R'\oplus L'$. We have $R'=ROL(R,\beta)\oplus L'$, so:
$S=(ROL(R,\beta)\oplus L')\oplus L' = ROL(R,\beta)$
But $\tilde{R}=ROR(S,\beta)=ROR(ROL(R,\beta),\beta)=R$, hence $\tilde{R}=R$.
Now consider $U:=((L'\oplus K)-\tilde{R})\bmod 2^{w}$. From $L'=((ROR(L,\alpha)+R)\bmod 2^{w})\oplus K$ we have $L'\oplus K \equiv_{2^w} ROR(L,\alpha)+R$, thus $U \equiv_{2^w} (ROR(L,\alpha)+R)-\tilde{R}$.
Since $R=\tilde{R}$, it results that $U \equiv_{2^w} ROR(L,\alpha)$.
Therefore, $\tilde{L}=ROL(U,\alpha)=ROL(ROR(L,\alpha),\alpha)=L \Rightarrow L=\tilde{L}$. Thus $g_K\circ f_K=id$.

*Proving $f_K(g_K(L',R'))=(L',R')$:*
Let $(\tilde{L},\tilde{R})=g_K(L',R')$, where $\tilde{R}=ROR(R'\oplus L',\beta)$ and $\tilde{L}=ROL(((L'\oplus K)-\tilde{R})\bmod 2^{w},\alpha)$.
Let $\tilde{L'}=((ROR(\tilde{L},\alpha)+\tilde{R})\bmod 2^{w})\oplus K$.
From the construction of $\tilde{L}$, we have $ROR(\tilde{L},\alpha) \equiv_{2^w} (L'\oplus K)-\tilde{R}$, thus $ROR(\tilde{L},\alpha)+\tilde{R} \equiv_{2^w} L'\oplus K$, so $\tilde{L'}=((L'\oplus K)\bmod 2^{w})\oplus K=L'$.
Now let $\tilde{R'}=ROL(\tilde{R},\beta)\oplus\tilde{L'}$. We have $\tilde{R}=ROR(R'\oplus L',\beta)$, so $ROL(\tilde{R},\beta)=R'\oplus L'$, thus $\tilde{R'}=(R'\oplus L')\oplus L'=R'$.
Therefore, $f_K(g_K(L',R'))=(L',R')$, meaning $f_K\circ g_K=id$.

## 3. Decryption Attack

The attacker observes ciphertexts encrypted in the same way without knowing the key used for their encryption. The goal is to find the last subkey used to encrypt a message. Once the last key is found, the process is repeated until the entire secret key is discovered. All evaluation metrics presented reflect the algorithms' ability to return this last subkey. The methodological benchmark is based on the approach proposed by Gohr, used for defining the problem, generating training data, and evaluating model performance.

### 3.1 Convolutional Neural Network (CNN)

#### Problem Definition
We construct a convolutional neural network (CNN) acting as a *neural distinguisher*. The network returns a probability $p \in [0, 1]$ to answer: "Does the ciphertext pair $(C_1, C_2)$ originate from the encryption of two plaintexts that adhere to a specific fixed difference?". Conversely, $1-p$ represents the probability that the pair consists of completely random bit sequences.

#### Training Data Generation
We use plaintext pairs that differ by a fixed XOR value, denoted $\Delta P = (\Delta L, \Delta R)$. Positive data generation involves encrypting these pairs:

$(L,R) \rightarrow C_1$
$(L\oplus \Delta L, R\oplus \Delta R) \rightarrow C_2$

A valid sample is formed by $(C_1, C_2)$. The dataset is balanced: half are positive examples, and the other half are negative examples (where $C_2$ is replaced with a uniformly random value).

#### Network Architecture
The architecture is a residual convolutional network consisting of a residual tower and a prediction head, optimized to reduce parameter count without accuracy loss.
* **Input:** 3D tensor with 3 channels, spatial dimension of 16 bits.
* **Residual Block:** The network integrates *depth* successive blocks containing `Conv1d`, `BatchNorm1d`, `ReLU`, and residual connections to prevent vanishing gradients.
* **Prediction Head:** Flattens the $(3, 16)$ output into a 48-element vector, passed through `Linear` layers, `BatchNorm1d`, `ReLU`, and finally a `Sigmoid` function.

#### Training Parameters and Strategy
Models are trained for 5, 6 (depth = 10), and 7 encryption rounds (depth = 1):
* **Loss Function:** `MSELoss`.
* **Optimizer:** `Adam`, $L2$ regularization, weight decay $10^{-5}$.
* **Learning Rate:** `OneCycleLR` scheduler, max rate $10^{-3}$.
* **Dataset:** $10^7$ total samples.
* **Batch Size:** 5000.
* **Duration:** 200 epochs.

Total training time for the 3 networks was approx. 64 hours using an i7 11th Gen CPU, GTX 1650 GPU (4GB VRAM), and 16GB RAM.

#### Training Results

| Model | Accuracy (%) |
| :--- | :--- |
| 5 rounds | 92.74 |
| 6 rounds | 78.79 |
| 7 rounds | 55.14 |
| 8 rounds | $\approx$ 50.00 |

Attempting to train 8 or more rounds with the same strategy yields $\approx$ 50% accuracy (equivalent to random guessing), making it unusable for the presented cryptanalysis algorithms.

### 3.2 Utilizing CNN Probabilities

The neural distinguisher (DND) is used as a core component in the subkey recovery phase (network inference on partially decrypted data).

#### Sum of Logits (SoL)
Since single-pair accuracy is limited, ciphertext structures based on neutral bits are used. CNN responses are aggregated to formulate a confidence score.
* $f_0(X) = P(real|X)$: CNN probability that $X$ is real.
* $X_i(K) = f^{-1}(C_i, K)$: partial decryption using candidate subkey $K$.
* $p_i(K) = f_0(X_i(K))$.
* $l_i(K) = \log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$: log-odds transformation.

**Score calculation:**
$$S(K) = \sum_{i=1}^{n}\log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$$

**SoL Attack Performance (Top 32)**

| Attack | Success Rate (%) | Average Real Key Rank |
| :--- | :--- | :--- |
| 6 rounds | 100.0 | 1.50 |
| 7 rounds | 100.0 | 1.60 |
| 8 rounds | 100.0 | 5.10 |

#### Bayesian Key Search (BKS)
When probing single-round decryption, the randomization hypothesis for wrong keys often fails. BKS addresses this using a Wrong Key Response Profile (WKRP).

**Iterative Search Algorithm:**
1. Start with a set $S$ of $n_{cand}$ candidate keys.
2. For each key $k_i \in S$, calculate log-odds $z_{j,k_i}$.
3. Calculate the average log-odds score $m_{k_i}$.
4. Calculate the Euclidean distance penalty score:
   $$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$
5. Update $S$ retaining the $n_{cand}$ keys that minimize $\lambda(k)$ and repeat.

**BKS Attack Performance (Top 32)**

| Attack | Success Rate (%) | Average Real Key Rank |
| :--- | :--- | :--- |
| 6 rounds | 100.0 | 1.00 |
| 7 rounds | 100.0 | 1.00 |
| 8 rounds | 20.0 | 2.50 |

### 3.3 Fine-Tuning the Model (Hard Examples)
To address performance drops during key recovery where erroneous decryptions do not produce perfectly random noise, we fine-tune the model using hard negative examples generated by encrypting an extra round with the correct subkey, then decrypting with a random subkey. This increased accuracy for such pairs by ~1%.

### 3.4 Proposed Models and Hybrid Algorithms
To maximize execution speed and attack success rate, several strategies were implemented:
* **Fine-Tuned BKS:** Executes BKS over the entire key space ($2^{16}$) exclusively using the FT model.
* **SoL $\rightarrow$ Original BKS:** Uses SoL to pre-filter to 64 candidates, then applies the original BKS.
* **SoL $\rightarrow$ Fine-Tuned BKS:** Uses SoL to pre-filter, followed by BKS using the FT model.
* **Ensemble BKS:** BKS inference is weighted between the standard and FT models using the inverse squares of the validation loss.
* **SoL $\rightarrow$ Ensemble BKS:** Filters 65536 keys down to 64 using SoL, then evaluates with Ensemble BKS.

### 3.5 BKS Algorithm Optimization
The WKRP generation is computationally expensive. We implemented offline caching, generating the tables ($\mu$ and $\sigma$ vectors) once per trained model and serializing them to disk. BKS only runs CNN inference on $n_{cand} = 64$ active candidate keys, evaluating the rest using vectorized tensor math via the cached WKRP, bypassing CNN calls entirely.

## 4. Evaluation and Results of the Cryptanalysis Methods

An automated testing framework simulates attacks on 6, 7, and 8 rounds using target subkeys and varying structural counts. Methods (M1-M6) are evaluated sequentially. A success is counted only if the correct key is ranked first. The evaluation loop halts when a proposed method (M2-M6) demonstrates clear dominance over the baseline (M1) either via superior accuracy or equivalent accuracy with strictly lower execution time.

### 4.1 Accuracy Analysis
For 6 and 7 rounds, M2, M4, and M5 quickly hit 100% accuracy, surpassing M1's 90%. For 8 rounds (an extreme case with massive noise), pure Bayesian methods (M1, M2, M5) recorded 0% success, whereas SoL pre-filtering methods (M3, M4, M6) successfully recovered the key in 20% of cases.

### 4.2 Execution Time Analysis
Direct BKS methods (M1, M2) finish under a second. M5 takes roughly double. Methods integrating SoL (M3, M4, M6) require CNN inference over 65536 keys, illustrating the speed-complexity trade-off.

**Results: 6 Rounds Attack (DND 5r)**

| Metric | M1 | M2 | M3 | M4 | M5 | M6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy (%) | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Avg Time (s) | 0.54 | 0.52 | 257.65 | 155.67 | 1.06 | 232.06 |

**Results: 7 Rounds Attack (DND 6r)**

| Metric | M1 | M2 | M3 | M4 | M5 | M6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy (%) | 90.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Avg Time (s) | 0.78 | 0.77 | 342.34 | 261.34 | 1.52 | 321.23 |

**Results: 8 Rounds Attack (DND 7r)**

| Metric | M1 | M2 | M3 | M4 | M5 | M6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy (%) | 0.0 | 0.0 | 20.0 | 20.0 | 0.0 | 20.0 |
| Avg Time (s) | 0.75 | 0.72 | 305.51 | 242.12 | 1.48 | 255.42 |

## 5. Conclusions and Future Research Directions

This paper demonstrates the efficiency of convolutional neural networks in the cryptanalysis of the Speck 32/64 cipher.

Fine-tuning on hard negative examples is essential for eliminating false correlations. Furthermore, offline caching of the WKRP profile eliminates the need for repeated inferences, making BKS vastly faster than standard SoL approaches. The primary contribution lies in the proposed hybrid algorithms (M2–M6) which provide an ideal compromise between speed and accuracy, successfully recovering keys even at 8 rounds where traditional models fail.

Future research directions include:
* **Exploring new neural architectures:** Transitioning from residual CNNs to attention-based mechanisms (Transformers) capable of capturing non-linear and global dependencies more efficiently.
* **Generalizing to other cryptographic primitives:** Adapting WKRP caching and hybrid algorithms to analyze other lightweight ciphers (e.g., Simon, ChaCha20, Simeck) to validate whether these vulnerabilities are ARX-specific or universally applicable.
