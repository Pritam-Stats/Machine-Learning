# SVM Kernels Explained

## **What is an SVM Kernel?**
Before talking about kernels, let's first understand **Support Vector Machines (SVM).**

### **What is SVM?**
- SVM is a supervised learning algorithm used for **classification** and **regression**.
- It works by finding the **best decision boundary (hyperplane)** that separates different classes in the dataset.
- The goal is to maximize the **margin** (distance between the hyperplane and the closest points, called **support vectors**).

### **The Problem: When Data is NOT Linearly Separable**
SVM works great when data is **linearly separable**, meaning we can draw a straight line (or hyperplane in higher dimensions) to separate the classes.

👉 **But what if the data is NOT linearly separable?**  
For example, consider this dataset:

| Feature 1 | Feature 2 | Class |
|-----------|-----------|--------|
| 1.2       | 2.5       | 0      |
| 2.3       | 3.1       | 0      |
| 3.5       | 4.7       | 1      |
| 5.2       | 6.1       | 1      |

If plotted, it may look like a circular or complex shape where a straight line cannot separate the data. This is where **kernels** help.

---

## **What Do Kernels Do?**
- Kernels **transform** the original data into a higher-dimensional space, where a linear separator **CAN** be found.
- Instead of working in the **original space**, we apply a mathematical function (**kernel function**) to map the data into a **new space**.

---

## **Types of SVM Kernels**
Let’s go step by step.

### **1. Linear Kernel**
- Used when data is **already** linearly separable.
- Example: A simple straight-line separator.
- Mathematically, it computes:

  $$K(x, y) = x \cdot y$$

- **Example Use Case**: Spam email detection, where words' frequency (features) can linearly separate spam and non-spam emails.

---

### **2. Polynomial Kernel**
- Useful when data has a **non-linear** relationship.
- Example: If data follows a parabolic shape.
- Equation:


  $$K(x, y) = (x \cdot y + c)^d$$

- $d$ is the polynomial degree (higher degree = more complex curves).

🛠 **Real-Life Use Case**: Image recognition (e.g., handwritten digit recognition) where the relationship is not purely linear.

---

### **3. RBF (Radial Basis Function) Kernel**
- The **most commonly used kernel**.
- It maps data into an **infinite-dimensional space** (imagine creating circular decision boundaries).
- Equation:

 $$ K(x, y) = \exp\left(-\frac{||x - y||^2}{2\sigma^2}\right)$$

- $\sigma$ controls how much influence each training point has.

🛠 **Real-Life Use Case**: Face recognition, where pixel values are mapped to a complex decision surface.

---

### **4. Sigmoid Kernel**
- Inspired by **neural networks**.
- It applies a **sigmoid activation function**:

$$K(x, y) = \tanh(\alpha x \cdot y + c)$$


- Not commonly used in practice but useful in some cases.

---

## **Visualizing Kernels**
To help understand, imagine data like this:

- **Linearly separable (Linear Kernel)**  
  🟥⬜⬜⬜ | 🟦🟦🟦🟦  
  *(A simple straight line can separate these classes.)*

- **Circularly separable (RBF Kernel)**  
  ⭕⬜⬜⬜⬜⭕  
  *(We need a curve to separate them.)*

- **Complex separable (Polynomial Kernel)**  
  ⬜⬜🟥⬜🟦⬜⬜  
  *(We need a more complex boundary.)*


