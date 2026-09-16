---
title: NetraX Clinical AI Screening Kiosk
emoji: 👁️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# sih_dr_project
# TECHFUSION / NetraX — Explainable AI for Diabetic Retinopathy Screening in Rural India

### Smart India Hackathon 2026

An explainable, human-in-the-loop AI screening and clinical decision-support system designed to enable accessible, reliable and scalable Diabetic Retinopathy (DR) screening in rural and resource-constrained healthcare settings.

---

## 🩺 Overview

Diabetic Retinopathy is a major cause of preventable vision loss, yet access to regular retinal screening remains limited in rural areas due to shortage of ophthalmologists, large screening populations, variable-quality fundus images and poor connectivity.

**TECHFUSION** addresses these challenges through an intelligent retinal screening pipeline that combines automated image quality assessment, retinal structure analysis, lesion detection, vascular graph analysis, explainable AI and confidence-based ophthalmologist escalation.

The system is designed to support healthcare workers at Primary Health Centres (PHCs) while keeping ophthalmologists in the clinical decision loop.

---

## 💡 Proposed Solution

The complete screening pipeline follows:

**Portable Fundus Camera → Image Quality Assessment → Enhancement → Retinal Structure & ETDRS Mapping → FNO + GNN/GAT Analysis → Feature Fusion → 5-Level DR Grading → Confidence Gate → Explainable Report → Referral / Monitoring**

Ungradable images are automatically rejected and sent for recapture before further analysis. Acceptable images are enhanced and processed for retinal structures, lesions and vascular abnormalities.

---

## 🏗️ System Architecture

.
              PORTABLE FUNDUS CAMERA
                       │
                       ▼
             IMAGE QUALITY ASSESSMENT
                       │
                 ┌─────┴─────┐
                 │           │
             Ungradable   Acceptable
                 │           │
                 ▼           ▼
              RECAPTURE   ENHANCEMENT
                             │
                             ▼
              RETINAL STRUCTURE ANALYSIS
              ├── Optic Disc
              ├── Fovea
              ├── Retinal Vessels
              └── ETDRS Mapping
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              FNO BRANCH         GNN / GAT BRANCH
              Lesion Analysis    Vascular Analysis
                    │                 │
                    └────────┬────────┘
                             ▼
                       FEATURE FUSION
                             │
                             ▼
                      5-LEVEL DR GRADING
                             │
                             ▼
                      CONFIDENCE GATE
                       ┌─────┴─────┐
                       ▼           ▼
                 High Confidence   Low / Conflicting
                       │           │
                       ▼           ▼
                 Report /        Ophthalmologist
                 Referral           Review
