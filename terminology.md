## Key Terminology Reference

### Phase 1-2 — NLP & Statistical Analysis

**Unigrams** — single words treated as individual features ("artery", "fracture").
Simple but lose context — "range" and "motion" are unrelated unigrams even though
"range of motion" is a meaningful medical phrase.

**Bigrams / Trigrams** — sequences of 2 or 3 consecutive words ("range motion",
"left ventricular ejection"). Capture the "procedural language" of medicine that
single words miss. Part of the `ngram_range=(1,3)` setting in TF-IDF.

**N-Grams** — the general term for any n-word sequence. Unigrams are n=1,
bigrams n=2, trigrams n=3.

**Top Words (Frequency Analysis)** — counting which words appear most often,
overall and per specialty. Identifies common vocabulary but cannot distinguish
*distinctive* words from *universal* ones ("patient" tops every list).

**Chi-Square (χ²) Test** — a statistical test asking: "is this word's appearance
*dependent* on the specialty, or independent (random)?" Compares observed word
frequency per specialty against what you'd expect if the word appeared randomly.
High χ² score = the word is statistically "locked" to a specialty. Proves "artery"
in Cardiology is not a fluke — it is a defining characteristic.
- H₀ (null): the word's occurrence is independent of specialty.
- H₁ (alternative): the word and specialty are dependent.

**PMI (Pointwise Mutual Information)** — measures whether two words co-occur
*more than chance predicts*. Separates meaningful medical phrases ("internal
carotid" — words that seek each other out) from random pairings ("the patient" —
two common words that collide by chance). High PMI = genuine term. Low PMI =
grammatical filler.

**TF-IDF (Term Frequency — Inverse Document Frequency)** — scores each word by
combining how much *this document* uses it (TF) with how *rare* it is globally
(IDF). Words common everywhere ("patient") score near zero because high TF is
cancelled by low IDF. Words rare globally but common locally ("lithotripsy" in
Urology) score high. Automatically filters "medical noise" without manual removal.

---

### Phase 3 — Machine Learning & Classification

**Vectorization** — converting raw text into numbers a model can process.
We use TF-IDF vectorization: each note becomes a row of TF-IDF scores, one per
vocabulary word. The result is a matrix of shape (notes × features).

**X_train, X_test** — the feature matrices (TF-IDF scores) for training and
testing respectively. X = the inputs the model reads.

**y_train, y_test** — the label arrays (specialty names) for training and testing.
y = the correct answers the model tries to predict.

**test_size** — the fraction of data held out for testing (0.20 = 20%). The model
never sees this data during training — it is the "final exam." Keeping it separate
is what makes the evaluation honest.

**Stratified Split** — a train/test split that preserves class proportions in both
sets. Without it, rare specialties might land entirely on one side. With
`stratify=y`, if Cardiovascular is 13% of the data, it is 13% of train AND 13%
of test. Essential for imbalanced datasets.

**Logistic Regression** — a linear classifier that learns a *weight* (coefficient)
for every word-feature, per specialty. To classify a note, it sums the weighted
word scores and picks the highest specialty. Fast, interpretable (you can read
the coefficients), and often the strongest performer on text.

**Multinomial Naive Bayes** — a probabilistic classifier that asks "given these
words, which specialty is most probable?" Assumes word independence (each word
contributes independently — the "naive" assumption). Works surprisingly well for
bag-of-words despite this simplification. Fastest model; no class weighting.

**Random Forest** — builds hundreds of decision trees, each on a random data
subset, then votes. Captures non-linear patterns that linear models miss. Often
underperforms on sparse TF-IDF data (many zero columns confuse tree splits) but
included to test whether complexity helps.

**Confusion Matrix** — a grid where rows = true specialty, columns = predicted
specialty. The diagonal = correct predictions. Off-diagonal cells = specific
confusions ("how many Neurology notes were predicted as Radiology?"). Normalized
by row so each cell shows the *proportion* of that true class, removing the
class-size effect.

**Macro F1** — averages the F1 score for each specialty *equally*, regardless of
class size. Surgery (1,087 notes) counts the same as Rheumatology (10 notes).
If Macro F1 is much lower than accuracy, the model is ignoring rare specialties.
The honest metric for imbalanced classification.

**Parametric Classifier** — a model that *learns parameters* (weights,
probabilities, tree splits) from training data. After training, it no longer
needs the data — the knowledge is stored in the parameters. All three Phase 3
models are parametric.

---

### Phase 4-5 — Semantic Search & Validation

**Embeddings** — representing text as coordinates in high-dimensional space
(384 dimensions for our model). Semantically similar text lands *near each other*
in this space — "Renal" and "Kidney" are nearby points; "Renal" and "Banana" are
far apart. Distance = dissimilarity of meaning.

**Non-Parametric Classifier** — a model that learns *no parameters*. Instead of
training, it stores the data and references it at query time. Remove the stored
data and it cannot predict. Our mini-RAG is non-parametric — it retrieves similar
notes and votes. Contrast with Phase 3's trained models which can predict without
the original data.

**Mini-RAG (Retrieval-Augmented Generation — simplified)** — embed a query →
retrieve the k most similar stored chunks → vote on their specialty labels.
Called "mini" because true RAG passes retrieved context to a language model that
*generates* an answer; our version replaces generation with a simple majority vote.

**Layperson / Synonym Test** — querying with everyday language that contains
*zero* exact medical keywords ("the bone in my lower leg snapped" instead of
"tibial fracture"). If the system correctly routes this to Orthopedic, it
demonstrates semantic understanding beyond keyword matching — something TF-IDF
cannot do because it sees zero word overlap with stored notes.

**Extra Holdout in Phase 4 (why not automatic?)** — Phase 3's train/test split
is automatic because `train_test_split` is designed for supervised models: one
set trains, one set tests. Phase 4 has no training, so there is no automatic
split. We must manually hold out notes *before indexing* — if we indexed all
notes and queried with the same ones, the system would trivially retrieve copies
of itself (self-match), making the validation meaningless. Phase 4's holdout is
not "extra" — it is the *only* way to get an honest evaluation of a retrieval
system.

**Precision@K (Precision@5)** — for one query, the fraction of the k retrieved
chunks that carry the correct specialty label. Example: true = Neurology,
retrieved = [Neurology, Radiology, Neurology, Neurology, Surgery] → 3/5 = 60%.
Averaged across all held-out notes to give the mean Precision@5 headline metric.
A random baseline across 26 classes ≈ 3.8% — our ~34% is ~9× better than chance.

**Retrieval Confusion Matrix** — same structure as Phase 3's confusion matrix but
for the search engine: query's true specialty (rows) vs. majority-vote of retrieved
chunks (columns). Off-diagonal hotspots = semantic overlaps — pairs of specialties
whose clinical language is genuinely similar. Not failures: they reflect real
multi-disciplinary medicine. If the same pairs confuse both Phase 3's trained
classifier and Phase 5's untrained retriever, the overlap is inherent to the data.