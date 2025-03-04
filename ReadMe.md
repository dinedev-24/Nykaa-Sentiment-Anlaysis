**Nykaa Sentiment Analysis Project**

**📌 Project Overview**

This project analyzes **Nykaa product reviews** to extract sentiment insights using: 

✅ **Traditional Machine Learning Models** (Logistic Regression, SVM, Random Forest, XGBoost).
✅ **Fine-Tuned BERT for Sentiment Analysis**.
✅ **Aspect-Based Sentiment Analysis (ASBA)** to evaluate **Price, Quality, Fragrance, Packaging**.

-----
**📊 Project Workflow**

1️. **Data Preprocessing & Cleaning**

- Removed duplicates, handled missing values.
- Cleaned text using regular expressions.
- Assigned sentiment labels based on review ratings.

2️. **Traditional Machine Learning Approach**

- Converted text into numerical form using **TF-IDF Vectorization**.
- Trained models: **Logistic Regression, SVM, Random Forest, XGBoost**.
- Evaluated models using **Accuracy, Precision, Recall, F1-score**.

3️. **BERT-Based Sentiment Analysis**

- Used **Hugging Face Transformers** to fine-tune bert-base-uncased on Nykaa reviews.
- Achieved **higher accuracy (77.7%)** than traditional ML models.
- Visualized **BERT's attention mechanism** for word importance.

4️. **Aspect-Based Sentiment Analysis (ASBA)**

- Extracted product aspects: **Price, Quality, Fragrance, Packaging**.
- Analyzed sentiment distribution across aspects.
- **Price & Quality were the most discussed aspects** in reviews.
-----
**📈 Model Performance Comparison**

|**Model**|**Accuracy**|**F1-Score**|
| :- | :- | :- |
|**Logistic Regression**|74\.6%|74\.5%|
|**SVM**|74\.0%|74\.0%|
|**Random Forest**|69\.2%|69\.0%|
|**XGBoost**|69\.7%|68\.5%|
|**BERT**|**77.7%**|**77.8%**|

✅ **BERT outperformed all traditional ML models** due to its superior contextual understanding. 

✅ **Aspect-Based Sentiment Analysis (ASBA) provided deep insights into customer concerns.**

-----
**📂 Project Structure**

📁 Nykaa-Sentiment-Analysis

│── 📜 Nykaa Sentiment Analysis.ipynb  # Code for ML & BERT models

│── 📜 Nykaa Project Report.docx       # Detailed project report

│── 📜 nyka\_top\_brands\_reviews.csv     # Processed dataset

│── 📜 README.md                        # Project documentation

-----
**🚀 How to Run the Project**

1️⃣ Clone this repository:

git clone https://github.com/dinedev-24/Nykaa-Sentiment-Analysis.git

2️⃣ Install dependencies:

pip install -r requirements.txt

3️⃣ Run Jupyter Notebook:

jupyter notebook Nykaa\_Sentiment\_Analysis.ipynb

-----
**📌 Next Steps & Improvements**

✅ **Deploy as a Web App:** Create a UI dashboard to visualize sentiment insights.
✅ **Real-Time Review Analysis:** Integrate live review fetching & prediction.
✅ **Enhance ASBA:** Expand aspect categories for deeper analysis.

-----
**🛠 Contributors**

👤 **Dinesh Kumar (dinedev-24)**
📌 **GitHub:** [Nykaa-Sentiment-Analysis](https://github.com/dinedev-24/Nykaa-Sentiment-Analysis)

-----
🎯 **This project provides valuable insights into Nykaa product sentiment and can help businesses optimize customer experience.** 🚀

