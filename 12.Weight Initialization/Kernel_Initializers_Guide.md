# Complete Guide to `kernel_initializer` in Keras / TensorFlow

---

## 1. What is the Default Value?

If you do not pass any value to `kernel_initializer` in Keras layers (such as `Dense`, `Conv2D`, `Conv1D`, etc.), the default value is:

$$\mathbf{\text{"glorot\_uniform"}}\quad \text{(also known as \textbf{Xavier Uniform Initializer})}$$

- **Class Equivalent:** `tf.keras.initializers.GlorotUniform()`
- **Distribution:** Draws samples from a uniform distribution within $[-limit, limit]$ where:
  $$limit = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$$
  *(where $\text{fan\_in}$ is the number of input units and $\text{fan\_out}$ is the number of output units)*.

---

## 2. Quick Reference Table: Which Initializer to Use?

| Activation Function | Recommended Initializer | String Alias | Class Name |
| :--- | :--- | :--- | :--- |
| **Sigmoid / Tanh / Softmax** | Xavier / Glorot | `'glorot_uniform'` *(default)*<br>`'glorot_normal'` | `GlorotUniform()`<br>`GlorotNormal()` |
| **ReLU / Leaky ReLU / PReLU** | He / Kaiming | `'he_normal'` *(recommended)*<br>`'he_uniform'` | `HeNormal()`<br>`HeUniform()` |
| **SELU** | LeCun | `'lecun_normal'`<br>`'lecun_uniform'` | `LecunNormal()`<br>`LecunUniform()` |
| **RNNs / Deep LSTMs** | Orthogonal | `'orthogonal'` | `Orthogonal()` |

---

## 3. All Available `kernel_initializer` Values in Keras

In Keras, you can pass initializers in two ways:
1. **String shortcut (alias)** — e.g., `kernel_initializer='he_normal'` (uses default parameters).
2. **Initializer class object** — e.g., `kernel_initializer=tf.keras.initializers.HeNormal(seed=42)` (allows setting custom parameters like `seed`).

---

### A. Variance Scaling Initializers (Most Common & Recommended)

#### 1. Glorot / Xavier Family
Best suited for symmetric, zero-centered or saturating activations like **Tanh**, **Sigmoid**, or **Softmax**.

- **`'glorot_uniform'`** (`tf.keras.initializers.GlorotUniform(seed=None)`) — **DEFAULT**
  - Uniform distribution in $[-\sqrt{6 / (\text{fan\_in} + \text{fan\_out})}, +\sqrt{6 / (\text{fan\_in} + \text{fan\_out})}]$
- **`'glorot_normal'`** (`tf.keras.initializers.GlorotNormal(seed=None)`)
  - Truncated normal distribution with mean $0$ and standard deviation:
    $$\sigma = \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$$

#### 2. He / Kaiming Family
Best suited for non-saturating activations like **ReLU**, **LeakyReLU**, **ELU**, **PReLU**.

- **`'he_normal'`** (`tf.keras.initializers.HeNormal(seed=None)`)
  - Truncated normal distribution with mean $0$ and standard deviation:
    $$\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$
- **`'he_uniform'`** (`tf.keras.initializers.HeUniform(seed=None)`)
  - Uniform distribution in $[-\sqrt{6 / \text{fan\_in}}, +\sqrt{6 / \text{fan\_in}}]$

#### 3. LeCun Family
Best suited for **SELU** (Scaled Exponential Linear Units) activation (self-normalizing networks).

- **`'lecun_normal'`** (`tf.keras.initializers.LecunNormal(seed=None)`)
  - Truncated normal distribution with $\sigma = \sqrt{1 / \text{fan\_in}}$
- **`'lecun_uniform'`** (`tf.keras.initializers.LecunUniform(seed=None)`)
  - Uniform distribution in $[-\sqrt{3 / \text{fan\_in}}, +\sqrt{3 / \text{fan\_in}}]$

#### 4. Generic Variance Scaling
- **`tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None)`**
  - Base class underlying Glorot, He, and LeCun.
  - `mode`: `'fan_in'`, `'fan_out'`, or `'fan_avg'`
  - `distribution`: `'truncated_normal'`, `'untruncated_normal'`, or `'uniform'`

---

### B. Standard Distribution Initializers

- **`'random_normal'`** (`tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)`)
  - Normal (Gaussian) distribution.
- **`'random_uniform'`** (`tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)`)
  - Standard uniform distribution.
- **`'truncated_normal'`** (`tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)`)
  - Normal distribution discarding values more than 2 standard deviations away from the mean (prevents extreme outlier weights).

---

### C. Constant Initializers (⚠️ Not recommended for kernels/weights)

> [!WARNING]
> Initializing all weights/kernels to constant values (like all 0s or all 1s) causes **symmetry problems** where all neurons compute the exact same gradient and learn the exact same feature. These are typically only used for `bias_initializer`.

- **`'zeros'`** (`tf.keras.initializers.Zeros()`) — All weights initialized to `0.0`.
- **`'ones'`** (`tf.keras.initializers.Ones()`) — All weights initialized to `1.0`.
- **`'constant'`** (`tf.keras.initializers.Constant(value=0.5)`) — All weights set to a fixed constant `value`.

---

### D. Special Structured Initializers

- **`'orthogonal'`** (`tf.keras.initializers.Orthogonal(gain=1.0, seed=None)`)
  - Initializes with a random orthogonal matrix.
  - Preserves vector norm across matrix multiplications; frequently used for **RNN recurrent weights** (`recurrent_initializer='orthogonal'`).
- **`'identity'`** (`tf.keras.initializers.Identity(gain=1.0)`)
  - Generates the 2D identity matrix. Only valid for 2D square matrices.

---

## 4. Usage Examples in Code

### Using String Aliases
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # Default (Glorot Uniform)
    layers.Dense(64, activation='tanh'),  # kernel_initializer="glorot_uniform" by default
    
    # He Normal for ReLU
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    
    # He Uniform for LeakyReLU
    layers.Dense(32, kernel_initializer='he_uniform'),
    layers.LeakyReLU(alpha=0.1),
    
    # LeCun Normal for SELU
    layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'),
    
    # Output layer
    layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
])
```

### Using Class Objects (With Custom Seeds & Parameters)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(
        units=128,
        activation='relu',
        kernel_initializer=tf.keras.initializers.HeNormal(seed=42),
        bias_initializer=tf.keras.initializers.Zeros()
    ),
    layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
    )
])
```

---

## 5. Custom Initializer (Advanced)

You can create your own initializer by writing a function or subclassing `tf.keras.initializers.Initializer`:

```python
import tensorflow as tf

class CustomRandomInitializer(tf.keras.initializers.Initializer):
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, shape, dtype=None):
        return tf.random.normal(shape, mean=0.0, stddev=self.factor, dtype=dtype)

    def get_config(self):  # Required for serialization
        return {'factor': self.factor}

# Usage:
# layers.Dense(64, kernel_initializer=CustomRandomInitializer(factor=0.05))
```
