## **Project Overview**

Using the [Medical Transcriptions Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), you will act as a Health Data Scientist to analyze clinical text, build a specialty classifier, and design a semantic search engine.

### **Phase 0: Foundations & Framing**

Before coding, define the domain. A **clinical transcription** is a formal record of a patient encounter, often dictated and transcribed. Unlike general text, it is dense with jargon, abbreviations, and shorthand.ok l

- **Challenges:** Severe class imbalance (some specialties have 10x more data than others), high dimensionality, and overlapping vocabulary (e.g., "pain" appears in almost every specialty).

## **Phase 1: EDA & NLP Foundations**

**Goal:** Clean the noise and understand the linguistic "fingerprint" of different medical fields.

### **1.1 Data Cleaning & Engineering**

- **Load & Filter:** Remove nulls in transcription and medical_specialty. Drop exact duplicates and extremely short notes (under 20 characters)
- **Feature Extraction:** Calculate character length, word count, and sentence count for every note.
- **Visualization:** \* Plot a bar chart of samples per specialty.
    - Create histograms of note lengths.
    - Use boxplots to compare note lengths across different specialties.

### **1.2 NLP Analysis**

- **Preprocessing:** Lowercase text, remove punctuation, and filter out standard English stopwords.
- **Frequency Analysis:** \* Identify top 20 words overall and per specialty.
    - Extract **Bigrams** (2-word phrases) and **Trigrams** (3-word phrases) to find specialty-specific terminology (e.g., "range of motion" in Orthopedics).

## **Phase 2: Statistical Significance & Linguistic Importance**

### **2.1 Chi-Square ($\\chi^2$) Test for Term Independence**

Instead of testing length, we test if specific words are statistically "locked" to a specialty.

- **The Setup:**
    - **Null Hypothesis ($H_0$):** The occurrence of a specific term (e.g., "Artery") is independent of the medical specialty.
    - **Alternative Hypothesis ($H_1$):** The term and the specialty are highly dependent.
- **The Test:** Use the Chi-Square test on your document-term matrix. This identifies the words that have the strongest mathematical "bond" to their category.
- **The Output:** Words with the highest $\\chi^2$ scores are your most powerful features. This proves that "Artery" appearing in Cardiology isn't a fluke; it's a defining characteristic.

### **2.2 N-Gram Analysis: Beyond Single Words**

Single words (unigrams) often lose context. Phrase analysis captures the "procedural language" of medicine.

- **Bigrams & Trigrams:** Analyze the frequency and significance of 2 and 3-word sequences.
- **The "Context" Proof:** \* The word "Internal" is generic.
    - The bigram **"Internal Carotid"** is a high-signal feature for **Neurology** or **Vascular Surgery**.
- **Statistical Filtering:** Use a "Pointwise Mutual Information" (PMI) score to find bigrams that occur together more often than they would by random chance. This separates meaningful medical terms (e.g., "Range of Motion") from random word pairings (e.g., "The Patient").

### **2.3 TF-IDF: The "Uniqueness" Metric**

TF-IDF is your most critical tool for Phase 2. It filters out the "medical noise."

- **The Logic:** \* **TF (Term Frequency):** "How much does Cardiology talk about _Stents_?"
    - **IDF (Inverse Document Frequency):** "Does everyone else talk about _Stents_ too?"
- **The Result:** A word like "Patient" gets a score near **zero** because it’s everywhere. A word like "Lithotripsy" gets a **huge** score in Urology because it’s rare globally but common locally.

## **Phase 3: Machine Learning & Classification**

### **3.1 Data Preparation Pipeline**

Before training, ensure your data is split correctly to handle the inherent class imbalance.

- **Stratified Train/Test Split:** Use an 80/20 split. Setting stratify=y is critical here; it ensures that rare specialties (like "Hospice") are represented in both the training and testing sets.
- **Vectorization Strategy:** Use **TF-IDF (Term Frequency-Inverse Document Frequency)**.
    - Set max_features between 5,000 and 10,000.
    - Set ngram_range=(1, 3) to capture both single words and important medical phrases like "blood pressure."

### **3.2 Model Training (The "Three-Tier" Approach)**

Train these three models to understand the trade-offs between speed, interpretability, and complexity:

1.  **Logistic Regression (The Linear Baseline):**
    - **Why:** It’s fast and provides "coefficients" which tell you exactly which words are driving a prediction.
    - **Tip:** Use class_weight='balanced' to help the model not ignore smaller specialties.
2.  **Multinomial Naive Bayes (The Text Specialist):**
    - **Why:** It assumes feature independence, which works surprisingly well for "bag-of-words" models. It is extremely robust against high-dimensional noise.
3.  **Random Forest or XGBoost (The Non-Linear Option):**
    - **Why:** To see if complex decision trees can capture relationships that linear models miss.
    - **Note:** These often require more tuning (like max_depth or n_estimators) to avoid overfitting on sparse TF-IDF data.

### **3.3 Evaluation & Model Comparison**

Don't rely on Accuracy alone. In a dataset with class imbalance, a model could be 80% accurate just by guessing the most common specialty every time.

- **Metric 1: Macro F1-Score:** This treats every specialty equally. If your Macro F1 is much lower than your Accuracy, your model is failing on the smaller specialties.
- **Metric 2: Confusion Matrix:** Visualize where the "leaks" are.
    - _Example:_ Are "Neurology" notes being misclassified as "Radiology" because both mention "MRI"?
- **Metric 3: Error Analysis:** Pick 5 notes where the model was "Confident but Wrong." Read the text—is the note too short? Is it actually a mislabeled record in the original dataset?

### **Comparison Framework**

Create a table in your notebook to summarize the results:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Model** | **Accuracy** | **Macro F1** | **Weighted F1** | **Best For...** |
| **Logistic Regression** | 0.XX | 0.XX | 0.XX | Interpretability & Speed |
| **Naive Bayes** | 0.XX | 0.XX | 0.XX | Handling Sparse Data |
| **Random Forest** | 0.XX | 0.XX | 0.XX | Capturing Non-linear Patterns |

### **Interpretability: The "Why" behind the prediction**

For your **Logistic Regression** model, extract the top 10 coefficients for a few specific specialties. This is the "Full Circle" moment where you prove your model learned real medicine.

- **Cardiology:** Does it prioritize "EKG," "artery," and "atrial"?
- **Orthopedics:** Does it prioritize "fracture," "joint," and "lateral"?

## **Phase 4: Vector DBs & Semantic Retrieval**

### **4.1 The Conceptual "Why"**

Traditional models (Phase 3) treat "Renal" and "Kidney" as two completely different entities. **Embeddings** represent words as mathematical coordinates in a high-dimensional space. In this space, "Renal" and "Kidney" sit right next to each other because they share a semantic context.

### **4.2 Step-by-Step Implementation**

#### **Step 1: The Chunking Strategy**

Clinical notes are often too long for a single embedding model to process effectively—the "meaning" of the beginning of the note gets lost by the end.

- **Action:** Write a function to split each transcription into chunks of **100–300 words**.
- **Overlap:** Optional but recommended. Use a 10% overlap (e.g., 20 words) between chunks so that context isn't cut off mid-sentence.
- **Metadata:** Crucial! Ensure each chunk carries its original medical_specialty label and a note_id.

#### **Step 2: Generating Dense Vectors**

- **Model Selection:** Use the SentenceTransformers library with the all-MiniLM-L6-v2 model. It strikes the best balance between speed and performance for a capstone project.
- **The Transformation:** Run your text chunks through the model to output a **dense vector** (usually 384 dimensions).
    - _Self-Check:_ Your final data structure should be a list of vectors, each paired with its specialty label.

#### **Step 3: Building the Vector Store**

You need an efficient way to calculate the "distance" between a user's query and thousands of medical chunks.

- **Tool Choice:** \* **FAISS:** Best for raw speed and local execution.
    - **ChromaDB:** Best if you want an easier API that handles metadata storage for you.
- **Indexing:** Load your vectors into the index. Use **Cosine Similarity** as your distance metric.

### **4.3 Building a "Mini RAG" Classifier**

Once the database is ready, you will create a **Non-Parametric Classifier**. This model doesn't "train" on the data; it just "references" it.

#### **The Retrieval Function**

1.  **Embed Query:** Take a user input (e.g., _"Patient has persistent cough and wheezing"_).
2.  **Search:** Use your Vector Store to find the **Top 5 (k=5)** most similar chunks from your dataset.
3.  **Vote:** Look at the medical_specialty of those 5 chunks.
    - _Example:_ If results are \[Pulmonology, Pulmonology, Internal Medicine, Pulmonology, Radiology\], the system predicts **Pulmonology**.

## **Phase 5: Retrieval Validation: The "Ground Truth" Test**

To check your vector system against the dataset's medical_specialty labels, implement the following evaluation workflow:

### **1\. The Validation Set-Up**

Pick 50–100 samples from your test set that the model has **never seen** during the indexing phase. For each sample:

- **The Query:** The transcription text.
- **The Label:** The actual medical_specialty assigned to that note.

### **2\. Metric: Top-K Accuracy (Precision @ K)**

Run each query through your Vector DB and retrieve the **Top 5 (k=5)** results.

- **Match Logic:** A retrieval is "Correct" if the retrieved note's specialty matches the query note's specialty.
- **Score:** Calculate the percentage of the 5 results that are correct.

**Example:** > \* **Query:** A note labeled "Neurology."

- **Top 5 Results:** \[Neurology, Radiology, Neurology, Neurology, Surgery\]
- **Precision @ 5:** 60% (3 out of 5 matched).

### **3\. The Retrieval Confusion Matrix**

Just like you did in Phase 3, you can build a Confusion Matrix for your Vector DB.

- Compare the **Query Label** vs. the **Majority Vote** of the retrieved notes.
- **Analysis:** If a "Gastroenterology" query consistently retrieves "Surgery" notes, you’ve identified a **Semantic Overlap**. This isn't necessarily a failure; it’s a reflection of how similar the language is in those two fields.

## **Final Deliverables**

1.  **EDA Notebook:** Visualizations of class distributions, note lengths, and n-grams.
2.  **ML Pipeline:** A clean workflow from raw text to evaluation metrics and confusion matrices.
3.  **Search Prototype:** A demonstration of the Semantic Search engine showing how it handles synonyms better than keywords.
4.  **Insight Presentation:** A short group presentation on the insights and output your team derived over the course of the capstone