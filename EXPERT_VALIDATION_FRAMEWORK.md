# 🎯 Sistem Validasi Expert Diplomasi - DTM Project

## Daftar Isi
1. [Overview](#overview)
2. [Framework Validasi](#framework-validasi)
3. [Proses Validasi](#proses-validasi)
4. [Ground Truth Dataset](#ground-truth-dataset)
5. [Fine-Tuning Strategy](#fine-tuning-strategy)
6. [Evaluasi Model](#evaluasi-model)

---

## Overview

Sistem validasi expert diplomasi dirancang untuk:
- ✓ Memvalidasi hasil Topic Modeling
- ✓ Memvalidasi hasil Stance Analysis
- ✓ Menciptakan Ground Truth dataset
- ✓ Fine-tuning model dengan expertise domain
- ✓ Mengukur model performance vs expert judgment

> Untuk panduan validasi khusus diplomasi / hubungan luar negeri, lihat `EXPERT_VALIDATION_DIPLOMACY.md`

---

## Framework Validasi

### 1. **Expert Panel Composition**

Validasi dilakukan oleh panel ahli yang terdiri dari:

| Role | Expertise | Tasks |
|------|-----------|-------|
| **Diplomat/Foreign Policy Expert** | International relations, diplomacy | Validate topic relevance & stance classification |
| **Political Analyst** | Political discourse, stance detection | Review topic coherence & political alignment |
| **Linguist** | Indonesian language, nuances | Verify text preprocessing & interpretation |
| **Subject Matter Expert** | Foreign policy domain | Domain-specific validation |

### 2. **Validation Rubric**

#### **Untuk Topic Modeling:**

| Dimensi | Skala | Kriteria |
|---------|-------|---------|
| **Relevance** | 1-5 | Seberapa relevan topik dengan kebijakan luar negeri? |
| **Coherence** | 1-5 | Seberapa koheren words/concepts dalam topic? |
| **Distinctiveness** | 1-5 | Seberapa distinct dari topics lain? |
| **Interpretability** | 1-5 | Seberapa mudah dipahami oleh expert? |
| **Completeness** | 1-5 | Apakah ada aspek penting yang terlewat? |

**Skor Pass:** ≥ 4 (Good) atau ≥ 3.5 (Acceptable)

#### **Untuk Stance Analysis:**

| Dimensi | Skala | Kriteria |
|---------|-------|---------|
| **Accuracy** | Correct/Incorrect | Apakah stance classification benar? |
| **Confidence** | 1-5 | Seberapa yakin expert dengan stance? |
| **Nuance** | 1-5 | Apakah model capture nuance/ambiguity? |
| **Context** | 1-5 | Apakah model understand context dengan baik? |

**Skor Pass:** ≥ 80% accuracy untuk stance classification

---

## Proses Validasi

### **Fase 1: Preparation**

1. **Sample Selection**
   - Random sampling: 10-15% dari total data
   - Stratified sampling: mencakup semua time period & topic
   - Minimum 100 samples untuk robust validation

2. **Expert Training**
   - Align dengan classification guidelines
   - Review sample cases bersama
   - Kesepakatan tentang gray areas

3. **Ground Truth Annotation**
   - Expert independently annotate samples
   - Inter-rater reliability check (Cohen's Kappa)
   - Target: Kappa ≥ 0.70 (Substantial agreement)

---

### **Fase 2: Validation Execution**

#### **Step 1: Topic Validation**

```
Untuk setiap Topic:
├── 1. Lihat top 10 words dalam topic
├── 2. Review sample documents (5-10 docs)
├── 3. Rate pada 5 dimensi (Relevance, Coherence, etc)
├── 4. Berikan expert feedback/notes
├── 5. Tentukan: VALID / NEEDS REVISION / INVALID
└── 6. Suggest label/interpretation
```

**Form Validasi Topic:**
```
Topic ID: [ID]
Top Words: [word1, word2, word3, ...]

Rating:
- Relevance: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Coherence: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Distinctiveness: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Interpretability: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Completeness: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]

Status: [ ] VALID  [ ] REVISION  [ ] INVALID

Suggested Label/Interpretation:
[Expert's interpretation of the topic]

Expert Comments:
[Additional feedback]

Expert Name: _________
Date: _________
```

#### **Step 2: Stance Validation**

```
Untuk setiap Comment:
├── 1. Baca komentar original
├── 2. Lihat model's prediction
├── 3. Tentukan stance sebenarnya (UNTUK/LAWAN/NETRAL)
├── 4. Rate confidence level
├── 5. Berikan feedback jika ada discrepancy
└── 6. Note: CORRECT / INCORRECT / AMBIGUOUS
```

**Form Validasi Stance:**
```
Comment ID: [ID]
Original Text: [Text]
Preprocessed Text: [Preprocessed]

Model Prediction: [POSITIVE/NEGATIVE/NEUTRAL] (Confidence: [X%])

Expert Judgment:
Stance: [ ] POSITIVE (PRO)  [ ] NEGATIVE (LAWAN)  [ ] NEUTRAL

Confidence Level:
[ ] Very Confident  [ ] Confident  [ ] Somewhat  [ ] Low  [ ] Ambiguous

Explanation:
[Why this is the correct stance]

Disagreement Reason (if different from model):
[If expert differs from model prediction]

Expert Name: _________
Date: _________
```

---

### **Fase 3: Analysis & Feedback**

1. **Compute Metrics**
   - Accuracy: % agreement antara model & expert
   - Kappa: Inter-rater agreement
   - Precision/Recall per class
   - Confusion matrix

2. **Error Analysis**
   - Identify common error patterns
   - Group errors by type
   - Prioritize improvement areas

3. **Generate Report**
   - Summary statistics
   - Error breakdown
   - Recommendation untuk model improvement

---

## Ground Truth Dataset

### **Dataset Structure**

```
ground_truth_topics.csv
├── topic_id
├── top_words_model
├── expert_label
├── relevance_score
├── coherence_score
├── status (VALID/REVISION/INVALID)
├── expert_notes
└── final_score

ground_truth_stance.csv
├── comment_id
├── text_original
├── text_preprocessed
├── model_prediction
├── model_confidence
├── expert_stance
├── expert_confidence
├── agreement (TRUE/FALSE)
├── error_type
└── expert_notes
```

### **Sample Ground Truth - Topics**

```csv
topic_id,top_words_model,expert_label,relevance_score,coherence_score,status
0,"kebijakan,diplomasi,bilateral,negara,kerjasama",Kerjasama Bilateral,5,5,VALID
1,"perdagangan,eksport,impor,ekonomi,tarif",Perdagangan Internasional,4,4,VALID
2,"krisis,konflik,zona,maritim",Zona Maritim/KTT,5,4,VALID
3,"iklim,energi,lingkungan,karbon",Isu Lingkungan & Energie,4,4,VALID
```

### **Sample Ground Truth - Stance**

```csv
comment_id,text_original,model_prediction,expert_stance,agreement,error_type
c001,"Setuju 100% dengan kebijakan ini!",POSITIVE,POSITIVE,TRUE,-
c002,"Ini sangat merugikan negara kita",NEGATIVE,NEGATIVE,TRUE,-
c003,"Hmm, sebagian setuju sebagian tidak",NEUTRAL,NEUTRAL,TRUE,-
c004,"Kebijakan ini akan menyelamatkan negara",NEGATIVE,POSITIVE,FALSE,Negation handling
```

---

## Fine-Tuning Strategy

### **1. Fine-Tuning Data Preparation**

```python
# Prepare dataset dari ground truth
fine_tune_data = {
    'posts': [
        {
            'text': 'original text',
            'topic': 'expert_label',
            'topic_id': expert_topic_id
        },
        ...
    ],
    'comments': [
        {
            'text': 'comment text',
            'stance': 'expert_stance',  # POSITIVE/NEGATIVE/NEUTRAL
            'confidence': expert_confidence
        },
        ...
    ]
}
```

### **2. BERTopic Fine-Tuning**

```python
# Update BERTopic dengan expert labels
expert_labels = ground_truth_df['expert_topic_id'].values

topic_model = BERTopic(...)
# Use expert labels sebagai seed untuk model initialization
topics, _ = topic_model.fit_transform(docs, expert_labels)
```

### **3. Stance Model Fine-Tuning**

```python
# Fine-tune sentiment model dengan expert annotations
from transformers import TextClassificationPipeline, TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./fine_tuned_stance_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## Evaluasi Model

### **1. Performance Metrics**

#### **Untuk Topic Modeling:**
```
Expert Agreement Rate: X%
Average Coherence Score: Y/5
Average Relevance Score: Z/5
Valid Topics: A/B (%)
```

#### **Untuk Stance Analysis:**
```
Overall Accuracy: X%
Precision (Per Stance):
  - Positive: X%
  - Negative: Y%
  - Neutral: Z%

Recall (Per Stance):
  - Positive: X%
  - Negative: Y%
  - Neutral: Z%

F1-Score: X
Cohen's Kappa: Y
```

### **2. Error Analysis Report**

**Top Errors:**
1. **Common Error Pattern 1** (#X occurrences)
   - Cause: [Root cause]
   - Impact: [Effect on results]
   - Recommended Fix: [Solution]

2. **Common Error Pattern 2** (#Y occurrences)
   - Cause: [Root cause]
   - Impact: [Effect on results]
   - Recommended Fix: [Solution]

---

## Quality Assurance Process

### **QA Checklist:**

- [ ] **Data Quality**
  - [ ] Sampling strategy documented
  - [ ] Sample size ≥ 100
  - [ ] Stratified sampling applied

- [ ] **Validation Process**
  - [ ] Expert panel trained
  - [ ] Inter-rater reliability ≥ 0.70
  - [ ] Validation completed
  - [ ] All forms signed & dated

- [ ] **Ground Truth**
  - [ ] Ground truth dataset created
  - [ ] All labels verified
  - [ ] Conflicts resolved via consensus
  - [ ] Dataset documented

- [ ] **Performance**
  - [ ] Model accuracy ≥ 75%
  - [ ] No systematic biases
  - [ ] Error analysis complete
  - [ ] Improvement recommendations provided

- [ ] **Documentation**
  - [ ] Process documented
  - [ ] Metrics reported
  - [ ] Limitations noted
  - [ ] Future improvements identified

---

## Implementation Timeline

**Week 1:** Expert Panel Setup
- Recruit experts
- Training & calibration
- Establish guidelines

**Week 2-3:** Validation Execution
- Phase 1: Topic validation
- Phase 2: Stance validation
- Inter-rater reliability check

**Week 4:** Analysis & Fine-Tuning
- Error analysis
- Ground truth dataset creation
- Model fine-tuning

**Week 5:** Evaluation & Reporting
- Performance evaluation
- Report generation
- Recommendations

---

## Expected Outcomes

✅ **Ground Truth Dataset**
- Validated topics (labeled)
- Validated stance annotations
- Ready for model evaluation & fine-tuning

✅ **Model Improvements**
- Better accuracy through fine-tuning
- Domain-aligned topic interpretations
- Better stance classification

✅ **Documentation**
- Detailed validation report
- Error analysis
- Recommendations for future improvements

✅ **Quality Metrics**
- Baseline performance
- Agreement rates
- Error patterns identified

---

## Success Criteria

| Criteria | Target | Status |
|----------|--------|--------|
| Expert Panel Recruited | 4-5 experts | [ ] |
| Training Completion | 100% | [ ] |
| Inter-rater Reliability | κ ≥ 0.70 | [ ] |
| Ground Truth Size | ≥ 300 samples | [ ] |
| Model Accuracy | ≥ 75% | [ ] |
| Agreement Rate | ≥ 80% | [ ] |
| Documentation Complete | 100% | [ ] |

---

## Repository untuk Expert Validation

### **Folder Structure:**
```
/validation/
├── expert_forms/
│   ├── topic_validation_form.md
│   ├── stance_validation_form.md
│   └── expert_instructions.md
├── raw_annotations/
│   ├── expert1_topics.csv
│   ├── expert1_stance.csv
│   ├── expert2_topics.csv
│   └── expert2_stance.csv
├── ground_truth/
│   ├── ground_truth_topics.csv
│   ├── ground_truth_stance.csv
│   └── consensus_data.csv
├── reports/
│   ├── validation_report.md
│   ├── error_analysis.md
│   ├── performance_metrics.csv
│   └── recommendations.md
└── fine_tuned_models/
    ├── topic_model_v2/
    ├── stance_model_v2/
    └── performance_comparison.csv
```

---

## Templat Expert Form

### **Template 1: Topic Validation Form**

```markdown
# FORM VALIDASI TOPIK
## Sistem Analisis Topik Dinamis - DTM Project

**Expert Name:** ________________  
**Date:** ________________  
**Organization:** ________________  

### Topic Information
- **Topic ID:** [X]
- **Top 10 Words:** [word1, word2, ...]

### Sample Documents
[5-10 sample documents akan ditampilkan]

### Evaluation

#### 1. Relevance terhadap Kebijakan Luar Negeri
Seberapa relevan topik ini dengan kebijakan luar negeri Indonesia?

[ ] 1 - Tidak relevan sama sekali
[ ] 2 - Sedikit relevan
[ ] 3 - Cukup relevan
[ ] 4 - Relevan
[ ] 5 - Sangat relevan & penting

#### 2. Coherence (Kohesi)
Seberapa koheren kata-kata dalam topik ini membentuk konsep?

[ ] 1 - Tidak koheren
[ ] 2 - Agak koheren
[ ] 3 - Cukup koheren
[ ] 4 - Koheren
[ ] 5 - Sangat koheren

#### 3. Distinctiveness (Keunikan)
Seberapa distinct topik ini dibanding topik lain?

[ ] 1 - Sangat overlap dengan topik lain
[ ] 2 - Banyak overlap
[ ] 3 - Cukup distinct
[ ] 4 - Distinct
[ ] 5 - Sangat distinct & unik

#### 4. Interpretability (Kemudahan Interpretasi)
Seberapa mudah Anda memahami maksud topik ini?

[ ] 1 - Sangat sulit dipahami
[ ] 2 - Sulit
[ ] 3 - Cukup mudah
[ ] 4 - Mudah
[ ] 5 - Sangat mudah

#### 5. Completeness (Kelengkapan)
Apakah ada aspek penting yang terlewat?

[ ] 1 - Banyak aspek penting terlewat
[ ] 2 - Ada beberapa aspek terlewat
[ ] 3 - Kurang lengkap
[ ] 4 - Cukup lengkap
[ ] 5 - Sangat lengkap

### Overall Assessment
**Status:** [ ] VALID  [ ] NEEDS REVISION  [ ] INVALID

**Suggested Official Label/Name:**
[Expert's interpretation of what this topic represents]

### Expert Comments
[Detailed feedback and suggestions]

---

**Pertanyaan Tambahan:**
1. Apakah topik ini mencerminkan diskusi tentang kebijakan luar negeri Indonesia?
2. Adakah interpretasi alternatif yang lebih baik?
3. Bagaimana topik ini relate dengan topik lain?
```

### **Template 2: Stance Validation Form**

```markdown
# FORM VALIDASI STANCE ANALYSIS
## Sistem Analisis Topik Dinamis - DTM Project

**Expert Name:** ________________  
**Date:** ________________  
**Comment Batch:** [Range]

### Comment Information
- **Comment ID:** [ID]
- **Original Text:** [Teks asli komentar]
- **Preprocessed Text:** [Teks setelah preprocessing]

### Model Prediction
- **Model Stance:** [POSITIVE / NEGATIVE / NEUTRAL]
- **Confidence:** [X%]

### Expert Judgment

#### Stance Classification
Menurut Anda, apa stance/sentimen dari komentar ini terhadap kebijakan luar negeri?

[ ] **POSITIVE (Mendukung/Setuju)**
    - Komentar ini mendukung atau setuju dengan kebijakan yang dibahas
    
[ ] **NEGATIVE (Menolak/Tidak Setuju)**
    - Komentar ini menolak atau tidak setuju dengan kebijakan
    
[ ] **NEUTRAL (Netral/Objektif)**
    - Komentar ini netral, informatif, atau tidak mengekspresikan dukungan/penolakan jelas

#### Confidence Level
Seberapa yakin Anda dengan klasifikasi di atas?

[ ] **Very Confident** - Sangat yakin, tidak ada keraguan
[ ] **Confident** - Yakin, stance cukup jelas
[ ] **Somewhat** - Fairly confident, ada konteks yang perlu pertimbangan
[ ] **Low Confidence** - Kurang yakin
[ ] **Ambiguous** - Ambigu, bisa diinterpretasi lebih dari satu cara

#### Agreement Check
Apakah penilaian Anda sama dengan model?

[ ] **AGREE** - Sama dengan model prediction  
[ ] **DISAGREE** - Berbeda dengan model prediction

**Jika DISAGREE, jelaskan:**
[Mengapa Anda berbeda dengan model prediction?]

### Additional Feedback
- Adakah nuansa yang tidak ditangkap model?
- Konteks apa yang perlu dipertimbangkan?
- Saran perbaikan?

---

**Signature:** ________________  
**Date:** ________________
```

---

## Next Steps

1. **Recruit Experts** - Hubungi diplomacy experts
2. **Setup Validation Process** - Setup forms & procedures
3. **Execute Validation** - Run validation dengan expert panel
4. **Create Ground Truth** - Compile hasil validasi
5. **Fine-tune Models** - Update models dengan ground truth
6. **Evaluate** - Measure improvement
7. **Document** - Report hasil & recommendations

---

**Status:** 📋 Framework Siap untuk Diimplementasikan
