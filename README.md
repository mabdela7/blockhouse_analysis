# Blockhouse Market Impact & Optimal Allocation Project

مشروع تحليل تأثير السوق المؤقت وتخصيص الأوامر المثلى

This project analyzes temporary market impact and formulates an optimal order allocation algorithm. It aims to demonstrate proficiency in data analysis, mathematical modeling, and clear communication of complex financial concepts.

يهدف هذا المشروع إلى تحليل تأثير التنفيذ المؤقت في السوق وبناء إطار رياضي وخوارزمي لتخصيص الأوامر المثلى عبر الزمن، مع توثيق شامل ثنائي اللغة.

## Project Structure | هيكلية المشروع

- `blockhouse_analysis.py`: Main Python script for data loading, impact modeling, and optimal execution.
- `impact_modeling_analysis.ipynb`: Jupyter Notebook containing the detailed analysis and visualizations.
- `data/`: Contains raw and extracted market data for CRWV, FROG, and SOUN tickers. جميع بيانات السوق منظمة في مجلدات فرعية لكل رمز: 
  - `data/CRWV/extracted/CRWV/`
  - `data/FROG/extracted/`
  - `data/SOUN/extracted/`
- `output/`: Stores generated plots and analysis results.
- `docs/`: Contains the HTML report and supplementary PDF documents.

## Key Features | الميزات الرئيسية

- **Data Processing | معالجة البيانات**: Handles loading and preprocessing of market data from various formats.
- **Temporary Impact Modeling | نمذجة التأثير المؤقت**: Implements linear and non-linear (power-law) models for temporary market impact `g_t(x)`.
- **Optimal Order Allocation | تخصيص الأوامر الأمثل**: Provides a mathematical framework and algorithm to determine optimal order allocation (`x_i`) over time intervals (`t_i`) to minimize trading costs, ensuring `Σx_i = S` (total shares).
- **Visualization | التصوير البياني**: Generates insightful plots to visualize impact models and optimal allocation strategies.
- **Comprehensive Documentation | توثيق شامل**: Includes detailed explanations in PDF format and an interactive HTML report.

## How to Run | كيفية التشغيل

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd blockhouse_analysis-main
    ```
2.  **Install dependencies | تثبيت المتطلبات:**
    ```bash
    pip install pandas numpy matplotlib seaborn scipy notebook nbconvert
    ```
3.  **Run the analysis script | تشغيل سكريبت التحليل:**
    ```bash
    python blockhouse_analysis.py
    ```
    This script will process data, fit models, and generate output plots in the `output/` directory.

4.  **View the detailed analysis (Jupyter Notebook) | استعرض التحليل المفصل:**
    ```bash
    jupyter notebook impact_modeling_analysis.ipynb
    ```
    You can also view the HTML version directly:
    `docs/impact_modeling_analysis_output.html`

5.  **Access Reports and Visualizations | الوصول للتقارير والرسوم:**
    Open `docs/index.html` in your web browser to navigate through the analysis report, plots, and supplementary documents.

## Deliverables | المخرجات

- **Interactive Analysis Report**: `docs/impact_modeling_analysis_output.html`
- **Modeling the Temporary Impact Function g_t(x)**: `docs/Modeling_the_Temporary_Impact_Function_g_t_x.pdf`
- **Mathematical Framework for Optimal Order Allocation**: `docs/Mathematical_Framework_for_Optimal_Order_Allocation.pdf`
- **Modeling Temporary Impact and Optimal Order Allocation (Presentation)**: `docs/Modeling_Temporary_Impact_and_Optimal_Order_Allocation.pdf`
- **Generated Plots**: Located in the `output/` directory and linked in `docs/index.html`.

## Contact | تواصل

For any questions or further discussion, please contact [Your Name/Email/LinkedIn].
للاستفسار أو النقاش، يرجى التواصل مع [اسمك/بريدك/لينكدإن].


