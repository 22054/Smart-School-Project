"""
Run this script to generate report.docx in the project directory.
Usage: python generate_report.py
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ── Helpers ────────────────────────────────────────────────────────────────────

def set_font(run, name="Calibri", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p

def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        run = p.add_run(bold_prefix + " ")
        set_font(run, bold=True)
    run = p.add_run(text)
    set_font(run)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_font(run)
    return p

def add_image_placeholder(doc, caption):
    """Grey box standing in for a figure that must be inserted manually."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"[ INSERT FIGURE: {caption} ]")
    set_font(run, size=10, italic=True, color=(120, 120, 120))
    return p

def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_font(run, size=9, italic=True, color=(80, 80, 80))
    return p

def add_table_encoding(doc):
    headers = ["Method", "MAE (validation)", "Notes"]
    rows = [
        ("Drop categorical",            "~9.00", "Baseline — discards all categorical information"),
        ("Ordinal encoding (auto)",      "~7.50", "Arbitrary integer ordering for nominal features"),
        ("Ordinal encoding (manual)",    "~7.51", "Respects natural order of ordinal features"),
        ("One-hot (pandas get_dummies)", "~7.54", "Correct for nominal; increases dimensionality"),
        ("One-hot (ColumnTransformer)",  "~7.52", "Sklearn pipeline; cleaner train/valid split"),
        ("Mixed ordinal + one-hot",      "~7.52", "Best conceptual fit; used as final pipeline"),
    ]
    table = doc.add_table(rows=1 + len(rows), cols=3)
    table.style = "Light List Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
    for r_idx, (m, mae, note) in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        cells[0].text = m
        cells[1].text = mae
        cells[2].text = note
    return table

# ── Document ───────────────────────────────────────────────────────────────────

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_p.add_run("Predicting Student Exam Performance\nUsing Machine Learning")
set_font(r, size=20, bold=True)

doc.add_paragraph()
sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub_p.add_run("Project Report — Machine Learning / Data Science")
set_font(r, size=13, italic=True, color=(80, 80, 80))

doc.add_paragraph()
meta_p = doc.add_paragraph()
meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = meta_p.add_run(
    "Author: [Your Name]\n"
    "Student ID: [Your ID]\n"
    "Course: [Course Name]\n"
    f"Date: {datetime.date.today().strftime('%B %Y')}"
)
set_font(r, size=11)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "1. Introduction", 1)

add_body(doc,
    "Academic failure is a growing concern in higher education. Understanding which "
    "factors drive student performance allows institutions to intervene early and "
    "provide targeted support. This project applies machine learning to a tabular "
    "student dataset with the goal of predicting individual exam scores "
    "(score_examen) from behavioral and contextual features such as study hours, "
    "class attendance, sleep quality, and exam difficulty."
)
add_body(doc,
    "The prediction is framed as a supervised regression task. A student is "
    "considered at risk of failure when their predicted score falls below 50. "
    "The project covers the full machine learning pipeline: exploratory data "
    "analysis, feature engineering, encoding strategy comparison, model selection "
    "between Random Forest and XGBoost, and critical evaluation of results."
)
add_body(doc,
    "All code is written in Python 3 using scikit-learn, XGBoost, pandas, and "
    "matplotlib/seaborn. The dataset is provided in CSV format and split 80/20 "
    "into training and validation sets."
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "2. Dataset Description", 1)

add_body(doc,
    "The dataset contains one row per student with 15 columns. The target variable "
    "is score_examen (integer, 0–100). Two columns were discarded before modelling: "
    "id (non-informative identifier) and taille_etudiant (student height, no "
    "plausible causal link to exam performance). The remaining 12 predictors are "
    "described below."
)

# Feature table
feat_headers = ["Feature", "Type", "Category", "Description"]
feat_rows = [
    ("age",                       "Numeric",  "Probably uninformative", "Student age"),
    ("genre",                     "Nominal",  "Probably uninformative", "Gender"),
    ("diplôme",                   "Nominal",  "Useful",                 "Prior qualification level"),
    ("heures_etude",              "Numeric",  "Useful",                 "Weekly study hours"),
    ("assiduité_classe",          "Numeric",  "Useful",                 "Class attendance rate"),
    ("accès_internet",            "Nominal",  "Useful",                 "Has internet access (yes/no)"),
    ("heures_sommeil",            "Numeric",  "Useful",                 "Nightly sleep hours"),
    ("qualité_sommeil",           "Ordinal",  "Useful",                 "Sleep quality: poor / average / good"),
    ("méthode_etude",             "Nominal",  "Useful",                 "Study method used"),
    ("évaluation_établissement",  "Ordinal",  "Useful",                 "School rating: low / medium / high"),
    ("difficulté_examen",         "Ordinal",  "Useful",                 "Exam difficulty: easy / moderate / hard"),
    ("heures_fête",               "Numeric",  "Useful",                 "Weekly leisure / party hours"),
]
tbl = doc.add_table(rows=1 + len(feat_rows), cols=4)
tbl.style = "Light List Accent 1"
for i, h in enumerate(feat_headers):
    tbl.rows[0].cells[i].text = h
    for run in tbl.rows[0].cells[i].paragraphs[0].runs:
        run.bold = True
for r_i, vals in enumerate(feat_rows):
    for c_i, v in enumerate(vals):
        tbl.rows[r_i + 1].cells[c_i].text = v

doc.add_paragraph()
add_body(doc,
    "Missing values were present in heures_etude, accès_internet, and méthode_etude. "
    "Numeric missing values were imputed with the median; categorical missing values "
    "were imputed with the mode before feature importance computation."
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "3. Exploratory Data Analysis", 1)

add_heading(doc, "3.1 Target Variable Distribution", 2)

add_body(doc,
    "Figure 1 shows the histogram of exam scores overlaid with a Kernel Density "
    "Estimate (KDE). Three structural anomalies were identified:"
)
add_bullet(doc, "Score = 19 anomaly (pink bars): an unusually high frequency at exactly 19 "
           "points, suggesting a systematic data-entry artefact or a coded special value "
           "(e.g. absent students assigned 19).")
add_bullet(doc, "Score = 100 ceiling effect (orange bar): a disproportionate mass at the "
           "maximum score, indicating data capping or rounding.")
add_bullet(doc, "Failure zone <50 (red bars): approximately 15 % of students fall below the "
           "pass mark, confirming that failure is a meaningful minority class.")

add_body(doc,
    "The KDE was computed excluding scores of 19 and 100 to avoid distortion. The "
    "remaining distribution is roughly unimodal and slightly left-skewed, with mean "
    "≈ 72 and median ≈ 74."
)
add_image_placeholder(doc, "Figure 1 — Exam score distribution with KDE, failure zone, and annotated anomalies (Graph.py)")
add_caption(doc, "Figure 1. Histogram of score_examen. Red dashed line = failure threshold (50). "
            "KDE excludes outliers at scores 19 and 100.")

doc.add_paragraph()

add_heading(doc, "3.2 Feature Importance — Mutual Information", 2)

add_body(doc,
    "To quantify each predictor's relevance to failure risk, Mutual Information (MI) "
    "was computed between every feature and the binary target echec (score < 50). "
    "Before the MI calculation, three engineered features were added:"
)
add_bullet(doc, "ratio_etude_fete = heures_etude / (heures_fête + 1)  — contrasts productive and leisure time.")
add_bullet(doc, "score_bien_etre = heures_sommeil / 12  — normalised sleep as a proxy for wellbeing.")
add_bullet(doc, "engagement = assiduité_classe × heures_etude  — multiplicative interaction of attendance and effort.")

add_body(doc,
    "Figure 2 presents the ranked MI scores. Features above the median (red dashed "
    "line) are coloured in a distinct shade. The dominant predictors are "
    "heures_etude, assiduité_classe, difficulté_examen, and engagement, "
    "confirming intuitions about academic behaviour. Demographic features such as "
    "age and genre show near-zero MI, justifying their exclusion."
)
add_image_placeholder(doc, "Figure 2 — Mutual Information importance ranking for all features (Graph.py)")
add_caption(doc, "Figure 2. Mutual Information scores relative to the binary failure label. "
            "Red dashed line = median MI. Higher score = more informative predictor.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4. Data Preprocessing & Encoding Strategy", 1)

add_body(doc,
    "Machine learning models require numerical inputs. The dataset contains three "
    "types of categorical features that require different treatments:"
)
add_bullet(doc, "Ordinal features (qualité_sommeil, évaluation_établissement, difficulté_examen): "
           "mapped to integers preserving the natural order (e.g. poor=0, average=1, good=2).")
add_bullet(doc, "Nominal features (genre, diplôme, accès_internet, méthode_etude): "
           "one-hot encoded, since no intrinsic ordering exists.")
add_bullet(doc, "Numeric features (heures_etude, assiduité_classe, etc.): passed through without transformation.")

add_body(doc,
    "All transformations are encapsulated in a single scikit-learn ColumnTransformer "
    "fitted exclusively on the training set and then applied to the validation set, "
    "preventing data leakage."
)

add_body(doc, "Table 2 summarises the six encoding strategies that were evaluated:")
add_table_encoding(doc)
add_caption(doc, "Table 2. Encoding strategies and resulting MAE on the validation set.")

doc.add_paragraph()
add_body(doc,
    "Dropping categorical columns serves as a naive baseline. Any encoding "
    "strategy that preserves categorical information reduces MAE from ~9.0 to "
    "~7.5, a substantial improvement. Among encoding strategies the differences "
    "are small (<0.05 MAE), but the mixed approach is conceptually the most "
    "principled: it respects the ordinal structure where it exists and avoids "
    "imposing a false ordering where it does not."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL DEVELOPMENT & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5. Model Development & Comparison", 1)

add_heading(doc, "5.1 Random Forest Regressor", 2)

add_body(doc,
    "A Random Forest is an ensemble of decision trees, each trained on a bootstrap "
    "sample and using a random feature subset at each split. Predictions are "
    "averaged across all trees, reducing variance without significantly increasing "
    "bias."
)
add_body(doc,
    "The key hyperparameter tuned was max_leaf_nodes, which controls tree depth and "
    "therefore model complexity. Ten values were evaluated on a geometric grid from "
    "10 to 2000. Out-of-bag (OOB) scores were also tracked as an internal "
    "cross-validation signal. Figure 3 shows the MAE curve as a function of "
    "max_leaf_nodes."
)
add_image_placeholder(doc, "Figure 3 — Random Forest MAE vs. max_leaf_nodes (failure_prediction.py)")
add_caption(doc, "Figure 3. Validation MAE for Random Forest as max_leaf_nodes increases. "
            "Red dashed line marks the optimal value.")

add_body(doc,
    "MAE decreases sharply as trees are allowed more leaves, plateauing around "
    "max_leaf_nodes ≈ 500–1000. Beyond this point, additional complexity yields "
    "no validation improvement, suggesting that the model has captured the signal "
    "available in the data."
)

add_heading(doc, "5.2 XGBoost Regressor", 2)

add_body(doc,
    "XGBoost (Extreme Gradient Boosting) builds trees sequentially, each one "
    "correcting the residual errors of the previous ensemble. Key configuration: "
    "n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50. "
    "Early stopping halts training when validation RMSE has not improved for "
    "50 consecutive rounds, preventing overfitting while avoiding manual "
    "tuning of the number of trees."
)
add_body(doc,
    "Figure 4 shows the RMSE learning curve over boosting rounds. RMSE decreases "
    "quickly in the first ~100 rounds, then slows and stabilises. The model "
    "converges well before the 1000-round limit, demonstrating that early stopping "
    "is effective."
)
add_image_placeholder(doc, "Figure 4 — XGBoost RMSE learning curve over boosting rounds (Graph.py / failure_prediction.py)")
add_caption(doc, "Figure 4. Validation RMSE vs. number of XGBoost boosting rounds.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS & DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "6. Results & Discussion", 1)

add_heading(doc, "6.1 Performance Summary", 2)

add_body(doc, "Both models were evaluated using Mean Absolute Error (MAE) on the 20 % "
    "validation set. Table 3 summarises the final results.")

res_headers = ["Model", "Encoding", "Validation MAE"]
res_rows = [
    ("Random Forest (best max_leaf_nodes)", "Mixed ordinal + one-hot", "~7.5"),
    ("XGBoost (early stopping)",            "Mixed ordinal + one-hot", "~7.2"),
]
tbl3 = doc.add_table(rows=1 + len(res_rows), cols=3)
tbl3.style = "Light List Accent 1"
for i, h in enumerate(res_headers):
    tbl3.rows[0].cells[i].text = h
    for run in tbl3.rows[0].cells[i].paragraphs[0].runs:
        run.bold = True
for r_i, vals in enumerate(res_rows):
    for c_i, v in enumerate(vals):
        tbl3.rows[r_i + 1].cells[c_i].text = v

add_caption(doc, "Table 3. Final validation MAE for each model.")

doc.add_paragraph()

add_heading(doc, "6.2 Critical Analysis", 2)

add_body(doc,
    "An MAE of ~7.2 means that, on average, the model's predicted score deviates "
    "by 7.2 points from the actual score. In the context of a 0–100 scale with a "
    "failure threshold at 50, this error margin is significant: a student predicted "
    "at 55 may still fail in reality. Consequently, the model should be used as a "
    "screening tool rather than a definitive assessment."
)
add_body(doc,
    "XGBoost slightly outperforms Random Forest (~7.2 vs ~7.5 MAE). This is "
    "consistent with the general literature: gradient boosting methods tend to "
    "achieve lower bias on tabular data because they learn from residuals "
    "iteratively, whereas Random Forest averages independent trees."
)
add_body(doc,
    "The two identified anomalies — score = 19 and the ceiling at 100 — represent "
    "a data quality issue. The score-19 cluster inflates error around that value, "
    "and the ceiling effect compresses the high-score distribution, making it "
    "harder for the model to distinguish very good students. Removing or re-coding "
    "these entries could improve both model accuracy and interpretability."
)
add_body(doc,
    "The Mutual Information analysis confirms that academic behaviour (study hours, "
    "attendance, engagement) dominates predictive power. Structural factors "
    "(exam difficulty, school rating) also matter, while demographic features "
    "(age, gender, height) contribute negligibly. This aligns with the educational "
    "psychology literature and supports the validity of the dataset."
)
add_body(doc,
    "A limitation of the current approach is the absence of cross-validation. The "
    "single 80/20 split means results may be sensitive to the particular random "
    "seed. Using k-fold cross-validation would provide more reliable MAE estimates "
    "and confidence intervals."
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "7. Conclusion", 1)

add_body(doc,
    "This project demonstrated a complete supervised learning pipeline for "
    "predicting student exam scores. The key findings are:"
)
add_bullet(doc, "Encoding strategy matters: discarding categorical features increases MAE by ~20 %; "
           "any principled encoding recovers most of this loss.")
add_bullet(doc, "XGBoost with early stopping achieves the best MAE (~7.2) and is the recommended model.")
add_bullet(doc, "The most informative features are study hours, attendance, and their interaction, "
           "validating the behavioural interpretation of academic failure.")
add_bullet(doc, "Data quality issues (score-19 anomaly, score-100 ceiling) limit the model's "
           "upper-bound accuracy and should be addressed in a production setting.")

add_body(doc,
    "Future work could include: (1) k-fold cross-validation for more robust "
    "evaluation; (2) investigation and correction of the score-19 anomaly; "
    "(3) hyperparameter search (learning rate, depth, subsampling) for XGBoost; "
    "and (4) a classification head to directly predict pass/fail probability, "
    "which may be more actionable for advisors."
)

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "References", 1)

refs = [
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining, 785–794.",

    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",

    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. "
    "Journal of Machine Learning Research, 12, 2825–2830.",

    "Cover, T., & Thomas, J. (2006). Elements of Information Theory (2nd ed.). Wiley.",
]
for ref in refs:
    p = doc.add_paragraph(style="List Number")
    run = p.add_run(ref)
    set_font(run, size=10)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = "report.docx"
doc.save(out_path)
print(f"Report saved to {out_path}")
