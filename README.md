# Hybrid-ML-Prediction

## Hybrid Machine Learning Model for smORFs Prediction in *Mycobacterium tuberculosis*

This repository contains supplementary materials and computational workflows supporting the manuscript:

**"Hybrid Machine Learning and Transcriptomics for Systematic Discovery of Bacterial smORFs"**

---

## Repository Structure

### 1. Machine Learning and Supplementary Data

* `Supplementary Materials/`
  Contains all supporting files for the manuscript:

  * `Mtb_positive_negative_data.csv` — Dataset for model training and evaluation
  * `Predicted_46_smORFs/`:

    * Supplementary File A - Predicted smORFs (FASTA)
    * Supplementary File B - Transcript abundances
    * Supplementary File C - BLAST comparison results
    * Supplementary File D - Overlap analysis (pathogenic vs non-pathogenic)
    * Supplementary File E - Comparison with previous studies
    * Supplementary File F - Functional annotation (Blast2GO)
    * Supplementary Table 1 - Bacterial genomes used
    * Supplementary Table 2 - Transcriptome dataset IDs
    * Supplementary Table 3 - Functional annotations
    * Supplementary Table 4 - Model performance metrics

* `hybrid_mtb_prediction.ipynb`
  Jupyter Notebook implementing the hybrid ML model

---

### 2. Transcriptome Analysis Pipeline

* `transcriptome_pipeline/`
  RNA-seq workflow used in this study

#### Pipeline Steps

* Quality control - FastQC
* Read preprocessing - Trimmomatic
* Alignment - BWA + Samtools
* Transcript assembly - StringTie
* Evaluation - Gffcompare

---

## How to Run the Pipeline

```bash
cd transcriptome_pipeline
bash pipeline.sh Read1.fastq.gz Read2.fastq.gz
```

---

## Requirements

Ensure the following tools are installed:

* FastQC
* Trimmomatic (v0.38)
* BWA
* Samtools
* StringTie
* Gffcompare

---

## Data Availability

Transcriptome raw data is available at:
https://drive.google.com/drive/folders/1q64BG9UpYWyoY8h-CVd22N9mGArnr8yV?usp=drive_link

---

## 📚 References

Cited references are included within the supplementary materials.
