# Fake News Detection System

## Overview
The **Fake News Detection System** is an innovative machine learning solution designed to identify whether a news article is real or fake. This model is trained using a **Decision Tree Classifier** and utilizes **TF-IDF Vectorization** for text preprocessing. It achieves a remarkable accuracy of **99.69%** on the test data, making it a reliable tool for detecting fake news in real-time.

The system takes a news article as input and classifies it into two categories:
- **Real News**
- **Fake News**

The web application powered by **Streamlit** offers an interactive and easy-to-use interface where users can input news articles and instantly receive a classification of the news content.

## Business Objective
In the age of misinformation, fake news poses a serious threat to public perception, political stability, and societal trust. The **Fake News Detection System** aims to:
1. **Enhance Information Credibility**: By providing an automated way to classify news articles, the system helps users quickly identify credible information from unreliable sources.
2. **Support Media and Journalism**: Journalists, media organizations, and fact-checking agencies can integrate this system to help sift through large volumes of information and focus on authentic news.
3. **Combat Misinformation**: By providing a tool for detection, this model can be used to reduce the spread of fake news on social media and news platforms.

## Accuracy
The model has achieved an accuracy of **99.69%** on the test data, demonstrating its ability to accurately classify news articles into real and fake categories. This is a significant improvement over previous methods and reflects the model's effectiveness in distinguishing between credible and misleading news.

## How the Model Works
The model is trained on a dataset containing real and fake news articles, which are processed and vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)**. After preprocessing the text, the **Decision Tree Classifier** is used to make predictions based on the input news article.

### Model Workflow:
1. **Preprocessing**: The text of the news article is cleaned by converting it to lowercase, removing URLs, punctuation, digits, and newline characters.
2. **Vectorization**: The text is converted into a numeric format using **TF-IDF Vectorizer**.
3. **Prediction**: The preprocessed and vectorized text is passed to the **Decision Tree Classifier** to predict if the article is real or fake.
4. **Output**: The prediction is returned as either "Real News" or "Fake News".

## Live Demo
You can test the system live by visiting the following link:  
[Fake News Detection Live Demo](https://fake-news-detection-none-system.streamlit.app/)

![Screenshot 2024-11-06 214827](https://github.com/user-attachments/assets/047239bc-b77d-49f1-8aa8-02c3eeabd75f)
![Screenshot 2024-11-06 214939](https://github.com/user-attachments/assets/97636e3d-275b-4a33-8356-5afbd4cf9b22)



## Business Use Cases
- **Media and Journalism**: Media organizations can use this system to quickly fact-check articles before publishing.
- **Social Media Platforms**: Platforms like Twitter, Facebook, or Reddit can integrate this model to flag or warn users about fake news.
- **Educational and Research Purposes**: Researchers and students can explore the model to understand machine learning techniques applied to real-world problems.

## Future Objectives
1. **Model Enhancement**: Explore the use of more complex models, such as **Random Forest**, **XGBoost**, or **Neural Networks**, to improve accuracy further.
2. **Scalability**: Integrate the system into larger platforms with higher data throughput and automatic news article scanning.
3. **Multilingual Support**: Extend the system to support multiple languages for global fake news detection.
4. **Real-time Detection**: Enable real-time detection for social media platforms to instantly flag potentially fake news articles as they are posted.

## app.py Explanation

The `app.py` file is the backbone of the **Fake News Detection System**'s web interface. It leverages **Streamlit**, an open-source framework for creating web applications, to provide an interactive user experience.

### Key Features of `app.py`:
- **Text Input**: The app allows users to input a news article by typing or pasting the text into a text area.
- **Text Preprocessing**: When a user submits an article, the text undergoes preprocessing using regular expressions to remove URLs, punctuation, digits, and other unnecessary characters.
- **Prediction**: After preprocessing, the article is vectorized using a **TF-IDF Vectorizer**, and the preprocessed text is passed to the **Decision Tree Classifier** model to predict whether the article is "Real News" or "Fake News".
- **Result Display**: The app then displays the result, providing feedback on whether the article is classified as real or fake.

## Installation

To run the system locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone


## Acknowledgements

We would like to acknowledge the following libraries and tools used in this project:

- **Scikit-learn**: For the tools to implement machine learning models like Decision Trees and TF-IDF vectorization.
- **Pandas**: For data manipulation and preprocessing of the dataset.
- **NumPy**: For numerical operations during data processing and model evaluation.
- **Streamlit**: For providing a simple framework to create interactive web applications.
- **Joblib**: For saving and loading the trained machine learning models.
- **GitHub**: For hosting the project and providing version control and collaboration features.

