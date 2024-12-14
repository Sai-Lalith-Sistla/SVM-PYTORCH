# SVM-PYTORCH

- Recommended Resources:
    - [StatQuest SVM YouTube video](https://www.youtube.com/watch?v=efR1C6CvhmE)
    - Udemy course [Learn Support Vector Machines in Python](https://www.udemy.com/course/machine-learning-adv-support-vector-machines-svm-python/?couponCode=KEEPLEARNING)
    - SVM section in *Hands-On Machine Learning* by Aurélien Géron.
    - PyTorch official tutorial: [PyTorch Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)


- Versions used in notebooks
    - !pip install torch --index-url https://download.pytorch.org/whl/cu124
    - Python version 3.10.11


## Concepts

#### Q : What is a hyperplane
    A hyperplane is a foundational concept in machine learning, particularly in the context of support vector machines (SVMs) and maximal margin classifiers. It's a flat subspace that divides an n-dimensional space (where n could be any number of dimensions, not just 2 or 3) into two parts. In a two-dimensional space, a hyperplane is a line; in three dimensions, it's a plane. In higher dimensions, it's a multidimensional separator that separates data points based on their characteristics or features.

#### Q : what are Support vectors
    Support vectors are the data points that lie closest to the decision surface


### Maximum Margin Classifier

The **Maximum Margin Classifier** is a machine learning algorithm that seeks to find the hyperplane that maximizes the margin between two classes. It is the foundation of Support Vector Machines (SVM) for binary classification and focuses on robustness by ensuring data points are as far as possible from the decision boundary.

#### **Pros and Cons**

| **Pros**                                | **Cons**                             |
|----------------------------------------|--------------------------------------|
| Effective for linearly separable data  | Struggles with non-linear boundaries |
| Robust to outliers near the margin     | Sensitive to noise in data           |
| Clear theoretical foundation           | Requires tuning for optimal results  |

The Maximum Margin Classifier serves as a conceptual stepping stone to understanding more complex SVM models with kernels for handling non-linear data.


### Support Vector Classifier (SVC)

The **Support Vector Classifier (SVC)** is a supervised machine learning algorithm used for classification tasks. It works by finding the hyperplane that best separates data points of different classes with the maximum margin. For non-linearly separable data, SVC applies kernel functions to transform the data into higher dimensions, enabling linear separability.

#### **Pros and Cons**

| **Pros**                                      | **Cons**                             |
|-----------------------------------------------|--------------------------------------|
| Handles both linear and non-linear data       | Computationally expensive for large datasets |
| Effective in high-dimensional spaces          | Sensitive to choice of kernel and parameters |
| Robust to overfitting (with proper tuning)    | Difficult to interpret results        |

SVC is a versatile and powerful algorithm for classification tasks, requiring careful parameter selection for optimal performance.



## Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks. It works by finding the optimal hyperplane that separates data points of different classes with the maximum margin. For non-linearly separable data, SVM uses kernel tricks to map data to a higher-dimensional space where it can be linearly separated.



### **Scenarios in Which SVM Can Be Used**
1. **Binary Classification**: Distinguishing between two categories, e.g., spam vs. non-spam emails.
2. **Multiclass Classification**: Using one-vs-one or one-vs-all strategies to classify multiple categories.
3. **Regression Problems**: Through Support Vector Regression (SVR), which minimizes error while maintaining a margin of tolerance.
4. **Image Recognition**: Classifying images based on features.
5. **Text Categorization**: Sentiment analysis, topic classification, etc.
6. **Bioinformatics**: Classifying genes or proteins.


## **Key Parameters of SVM**
1. **C (Regularization Parameter)**
   - Balances maximizing the margin and minimizing classification errors.
   - A smaller C prioritizes a larger margin (soft margin), allowing some misclassifications.
   - A larger C prioritizes correct classification over a large margin.

2. **Kernel**
   - Determines the type of decision boundary used in the higher-dimensional space.
   - Common kernels:
     - Linear: For linearly separable data.
     - Polynomial: For data with polynomial relationships.
     - RBF (Radial Basis Function): For non-linear data.
     - Sigmoid: Similar to neural network activation functions.

3. **Gamma**
   - Applicable for RBF and Polynomial kernels.
   - Controls the influence of individual data points.
   - Higher gamma focuses on nearby points, leading to tighter decision boundaries.
   - Lower gamma considers a broader range of points.

4. **Degree**
   - Relevant for the Polynomial kernel.
   - Specifies the degree of the polynomial decision boundary.

5. **Epsilon** (in SVR)
   - Defines a margin of tolerance around the predicted value in regression tasks.

6. **Shrinking**
   - Boolean parameter to enable or disable shrinking heuristics, which can speed up computation.



## **Pros and Cons**

| **Pros**                                | **Cons**                             |
|----------------------------------------|--------------------------------------|
| Effective in high-dimensional spaces   | Computationally expensive for large datasets |
| Works well with non-linear boundaries  | Requires careful tuning of parameters |
| Kernel flexibility for complex problems | Performance can degrade with noisy data |
| Robust to overfitting (with proper C)  | Difficult to interpret results        |
| Theoretical guarantees on performance  | Not ideal for very large datasets     |

---

SVM is a powerful algorithm suitable for a wide range of applications but requires thoughtful preprocessing and parameter tuning for optimal performance.

