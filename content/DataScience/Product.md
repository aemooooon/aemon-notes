---
title: 
draft: false
tags: 
date:
---
### 1. **Scalar Multiplication**

#### Elements Involved:
- A **scalar** (a single number) and a **vector** or **matrix**.

#### How to Calculate:
- Multiply **each element** of the vector or matrix by the scalar.

#### Example (with a vector):
Let $\vec{v}=\begin{bmatrix} 2 \\ 4 \end{bmatrix}$ and the scalar be $3$:
$$
3 \times \vec{v} = 3 \times \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 12 \end{bmatrix}
$$

#### Result:
- The result is a **vector** or **matrix** of the same size as the original, but each element is multiplied by the scalar.

---

### 2. **Dot Product (or Inner Product)**

#### Elements Involved:
- Two **vectors** of the same size.

#### How to Calculate:
- Multiply the corresponding elements of the vectors together and then sum the results.

#### Example (with two 3D vectors):
Let $\vec{a}=\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\vec{b}=\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$:
$$
\vec{a} \cdot \vec{b} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32
$$

#### Result:
- The result is a **scalar** (a single number).
- The dot product measures the alignment of two vectors: if the result is 0, the vectors are orthogonal.

---

### 3. **Matrix Multiplication**

#### Elements Involved:
- Two **matrices**, but their dimensions must satisfy the following condition: if matrix $A$ is of size $m \times n$, matrix $B$ must be of size $n \times p$. In other words, the **number of columns** of the first matrix must equal the **number of rows** of the second matrix.

#### How to Calculate:
- Multiply the elements of each row of the first matrix by the corresponding elements of each column of the second matrix, and then sum the results for each pair. The element in the $i^{th}$ row and $j^{th}$ column of the result matrix is the dot product of the $i^{th}$ row of the first matrix and the $j^{th}$ column of the second matrix.

#### Example:
Let $A=\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B=\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$:
$$
A \times B = \begin{bmatrix} 
(1 \times 5 + 2 \times 7) & (1 \times 6 + 2 \times 8) \\
(3 \times 5 + 4 \times 7) & (3 \times 6 + 4 \times 8) 
\end{bmatrix}
= \begin{bmatrix} 
19 & 22 \\
43 & 50
\end{bmatrix}
$$

#### Result:
- The result is a **matrix** with dimensions determined by the **rows of the first matrix** and the **columns of the second matrix**. In this case, $A$ is $2 \times 2$ and $B$ is $2 \times 2$, so the result is a $2 \times 2$ matrix.

---

### 4. **Cross Product**

#### Elements Involved:
- Two **3-dimensional vectors**.

#### How to Calculate:
- The cross product of two vectors $\vec{a}$ and $\vec{b}$, written as $\vec{a} \times \vec{b}$, is a new vector that is perpendicular to both $\vec{a}$ and $\vec{b}$, following the right-hand rule.
- For $\vec{a}=\begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix}$ and $\vec{b}=\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$, the cross product is:
$$
\vec{a} \times \vec{b} = \begin{bmatrix} 
a_2 b_3 - a_3 b_2 \\
a_3 b_1 - a_1 b_3 \\
a_1 b_2 - a_2 b_1
\end{bmatrix}
$$

#### Example:
Let $\vec{a}=\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\vec{b}=\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$:
$$
\vec{a} \times \vec{b} = \begin{bmatrix} 
(2 \times 6 - 3 \times 5) \\
(3 \times 4 - 1 \times 6) \\
(1 \times 5 - 2 \times 4)
\end{bmatrix}
= \begin{bmatrix} 
-3 \\
6 \\
-3
\end{bmatrix}
$$

#### Result:
- The result is a new **3-dimensional vector** that is perpendicular to the original vectors. The direction of the resulting vector is determined by the right-hand rule.

---

### Summary:

| Operation               | Elements Involved                      | How to Calculate                                        | Result Type            |
|-------------------------|----------------------------------------|--------------------------------------------------------|------------------------|
| **Scalar Multiplication**| Scalar and Vector/Matrix               | Multiply each element by the scalar                    | Vector or Matrix       |
| **Dot Product**          | Two Vectors (same size)                | Multiply corresponding elements and sum                | Scalar (single number) |
| **Matrix Multiplication**| Two Matrices (compatible dimensions)   | Multiply rows by columns and sum                       | Matrix                 |
| **Cross Product**        | Two 3D Vectors                        | Use the formula for each component of the new vector    | Vector (3D)            |

