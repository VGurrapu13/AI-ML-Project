# AI-ML-Project
Content Based Recommendation System 

## 📌 Overview
This project is a **content-based recommendation system** that suggests books based on a short text description provided by the user. It utilizes **TF-IDF vectorization** and **cosine similarity** to find books with the most relevant descriptions.

## 🗂 Dataset
The dataset (`books_data.csv`) contains:
- **Title**: Name of the book.
- **Author(s)**: Name of the author(s).
- **Avergae rating**: Book rating.

## ⚙️ Running of Code
Open the notebook file in Jupyter and the code runs as follows:
1. The user inputs a **description** of the type of books they like.
2. The system **converts book descriptions into numerical vectors** using **TF-IDF**.
3. It computes **cosine similarity** between the user's input and book descriptions.
4. The **top 5 most similar books** are displayed along with similarity scores.
5. The user can **continue entering inputs** for more recommendations.
6. Before exiting, the system asks for the user's **salary expectation** (as required in the task).

## 🛠 Setup Instructions

### **1️⃣ Install Dependencies**
Ensure you have **Python 3.9** installed, then install dependencies:

```sh
pip install pandas scikit-learn numpy
