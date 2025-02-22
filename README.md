# Diabetes-Risk-Prediction.
# Early-Stage Diabetes Risk Prediction

## 📌 Overview
Diabetes is a growing health concern worldwide, and early detection is crucial to prevent severe complications. This project leverages **machine learning** to predict early-stage diabetes risk based on medical parameters. The model achieves an **accuracy of 98%**, making it highly reliable for healthcare applications.

## 🚀 Features
✅ Predicts early-stage diabetes risk using medical data  
✅ **98% accuracy** achieved using a **Random Forest Classifier**  
✅ Interactive **Gradio UI** for user-friendly predictions  
✅ Fully **automated pipeline** from data preprocessing to prediction  
✅ Can assist **doctors and healthcare professionals** in early diagnosis  

## 🔬 Problem Statement
Diabetes cases are increasing, and many people remain undiagnosed until the disease progresses. Traditional diagnosis requires multiple tests and time. This project provides an **AI-powered alternative** that can predict diabetes risk **quickly and accurately** using machine learning.

## 🎯 Objectives
- Develop a **high-accuracy machine learning model** for diabetes prediction.
- Identify the **most important medical indicators** contributing to diabetes risk.
- Compare different **machine learning algorithms** to find the best-performing model.
- Create an **interactive UI** using **Gradio** for easy user access.

## 🏗️ Project Structure
```
Diabetes_Prediction/
│── main.py                  # Backend: Loads model & handles predictions and Frontend: User Interface for input & output
│── train_model.py           # Trained Machine Learning Model
│── dataset/                 # Contains diabetes_data.csv
│── README.md                # Project documentation
```

## ⚙️ Technologies Used
🔹 **Python** - Programming language  
🔹 **pandas, numpy** - Data processing  
🔹 **matplotlib, seaborn** - Data visualization  
🔹 **scikit-learn** - Machine learning model  
🔹 **Gradio** - UI for real-time predictions  
🔹 **Google Colab** - Model training & execution  

## 📊 Algorithms Used
✔ **Random Forest Classifier** (Main model with 98% accuracy)  
✔ **Decision Tree** (Used for comparison)  
✔ **Logistic Regression** (Baseline model)  
✔ **Support Vector Machine (SVM)** (Alternative approach)  

## 🖥️ How to Run the Project
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/Diabetes_Prediction.git
cd Diabetes_Prediction
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Backend and Frontend(Prediction API)**
```bash
streamlit run app.py
```


This will generate a link where you can enter patient data and get predictions.

## 📌 Results and Impact
- **Achieved 98% accuracy**, making it highly effective for early-stage diabetes prediction.
- Helps in **early detection**, reducing manual effort for healthcare professionals.
- Identifies **key medical factors** contributing to diabetes risk.


## 🙌 Contributors
- **Janhavi Sudake** - Problem Research, Frontend & Backend Development, Conclusion  
- **Disha Jagtap** - Algorithm Selection, Frontend & Backend Development, Results & Impact  

## ⭐ Feedback & Contributions
We welcome feedback and suggestions! If you’d like to contribute, feel free to open an issue or submit a pull request.  

🌟 **If you found this project useful, please give it a ⭐ on GitHub!**

