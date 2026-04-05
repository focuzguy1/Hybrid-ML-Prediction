#!/bin/bash

set -euo pipefail

########################################
#  Transcriptomics Pipeline
# Author: Babalola Abdulhafeez Oluwabunmi
# Purpose: workflow
########################################


# HOW TO RUN THIS SCRIPT
# bash pipeline.sh Read1.fastq.gz Read2.fastq.gz
#
# Example:
# bash pipeline.sh sample_R1.fastq.gz sample_R2.fastq.gz
#



# INPUT VALIDATION

if [ "$#" -ne 2 ]; then
    echo "ERROR: Please provide R1 and R2 FASTQ files"
    echo "Usage: bash pipeline.sh Read1.fastq.gz Read2.fastq.gz"
    exit 1
fi

READ1=$1
READ2=$2


# REFERENCES & TOOLS

REF_FA="index/GCF_000195955.2_ASM19595v2_genomic.fna"
REF_GFF="index/GCF_000195955.2_ASM19595v2_genomic.gff"

TRIMMOMATIC="trimmomatic-0.38.jar"
ADAPTERS="TruSeq3-PE.fa"

THREADS=8


# DIRECTORIES

mkdir -p qc trimmed aligned stringtie eval logs

echo "Pipeline started at $(date)" | tee logs/pipeline.log


# STEP 1: QC & TRIMMING

echo "STEP 1: QC & Trimming" | tee -a logs/pipeline.log

fastqc $READ1 $READ2 -o qc/

java -jar $TRIMMOMATIC PE -phred33 \
$READ1 $READ2 \
trimmed/R1_paired.fastq trimmed/R1_unpaired.fastq \
trimmed/R2_paired.fastq trimmed/R2_unpaired.fastq \
ILLUMINACLIP:${ADAPTERS}:2:30:10 \
SLIDINGWINDOW:8:30 MINLEN:50

fastqc trimmed/R1_paired.fastq trimmed/R2_paired.fastq -o qc/

echo "QC completed" | tee -a logs/pipeline.log

# STEP 2: ALIGNMENT
echo "STEP 2: Alignment" | tee -a logs/pipeline.log

if [ ! -f "${REF_FA}.bwt" ]; then
    bwa index $REF_FA
fi

bwa mem -t $THREADS $REF_FA \
trimmed/R1_paired.fastq trimmed/R2_paired.fastq | \
samtools sort -@ $THREADS -o aligned/reads.sorted.bam

samtools index aligned/reads.sorted.bam

echo "Alignment completed" | tee -a logs/pipeline.log

# STEP 3: TRANSCRIPT ASSEMBLY

echo "STEP 3: StringTie" | tee -a logs/pipeline.log

stringtie -p $THREADS \
-G $REF_GFF \
-o stringtie/assembly.gtf \
aligned/reads.sorted.bam

stringtie -e -B -p $THREADS \
-G $REF_GFF \
-o stringtie/abundance.gtf \
aligned/reads.sorted.bam

echo "StringTie completed" | tee -a logs/pipeline.log


# STEP 4: EVALUATION
echo "STEP 4: Evaluation" | tee -a logs/pipeline.log

gffcompare -r $REF_GFF -G \
-o eval/assembly_eval \
stringtie/assembly.gtf

echo "Evaluation completed" | tee -a logs/pipeline.log

echo "PIPELINE COMPLETED SUCCESSFULLY at $(date)" | tee -a logs/pipeline.log
