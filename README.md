# Offensive Yoruba Language Detection App

This repository contains a machine learning-based application for detecting offensive language in yoruba text. The app is built using either **Streamlit** or **Flask** and leverages a **Logistic Regression** model trained on a dataset of tweets. The model uses **TF-IDF Vectorization** for text preprocessing and classification.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Development](#model-development)
6. [File Structure](#file-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

The goal of this project is to detect if a lanuange in yoruba is based off of text inputs. The app takes a sentence as input and predicts whether it contains offensive language, hate speech, or is normal. The model is trained on a dataset of tweets and uses **TF-IDF Vectorization** for feature extraction and **Logistic Regression** for classification.

---

## Features

- **Text Input**: Users can input a sentence to check for offensive language.
- **Real-Time Prediction**: The app provides instant predictions using a pre-trained machine learning model.
- **Clean and Preprocess Text**: The app cleans and preprocesses the input text (e.g., removes emojis, URLs, and special characters) before making predictions.
- **Streamlit and Flask Support**: The app can be deployed using either Streamlit or Flask.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DominionAkinrotimi/Yoruba-Offensive-Language-Detection-Model.git
   cd offensive-language-detection
   ```

2. **Create a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   The app uses NLTK for text preprocessing. Download the required NLTK data by running:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

---

## Usage

### Streamlit App

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Open the App**:
   The app will open in your default web browser at `http://localhost:8501`.

3. **Enter Text**:
   Input a sentence in the text box and click **Predict** to see the result.

### Flask App

1. **Run the Flask App**:
   ```bash
   python app.py
   ```

2. **Open the App**:
   The app will be available at `http://localhost:5000`.

3. **Enter Text**:
   Input a sentence in the text box and click **Predict** to see the result.

---

## Model Development

The model was developed using the following steps:

1. **Data Cleaning**:
   - Convert text to lowercase.
   - Remove emojis, URLs, hashtags, mentions, and special characters.
   - Remove digits and extra spaces.

2. **Text Preprocessing**:
   - Tokenize the text.
   - Remove stopwords.

3. **Feature Extraction**:
   - Use **TF-IDF Vectorization** to convert text into numerical features.

4. **Model Training**:
   - Train a **Logistic Regression** model on the preprocessed data.

5. **Model Saving**:
   - Save the trained model and vectorizer using `joblib`.

---

## File Structure

```
offensive-language-detection/
├── app.py                  # Streamlit/Flask app
├── requirements.txt        # List of dependencies
├── logistic_regression_model.pkl  # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer
├── README.md               # Project documentation
├── templates/              # Flask HTML templates
│   └── index.html          # Flask app homepage
└── notebooks/              # Jupyter notebooks for model development
    └── model_development.ipynb
```

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset used for training was sourced from [Kaggle](https://www.kaggle.com).
- Special thanks to the developers of **scikit-learn**, **Streamlit**, and **Flask** for their amazing libraries.

---

## Contact

For questions or feedback, please contact:
- **Dominion Akinrotimi**  
- **Email**: akinrotimioyin@gmail.com  


---

Enjoy using the Offensive Yoruba Language Detection App! 🚀
