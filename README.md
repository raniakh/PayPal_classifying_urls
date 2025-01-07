# PayPal Product Web Page Classification

## Overview
This project focuses on a binary classification task: identifying whether a given web page is a product page or not. The classification leverages a combination of URL-based patterns, keyword-based content analysis, and embeddings to create a funnel solution for accurate classification with minimal computational resources.

## Motivation
This project was developed as part of a 5-month internship at PayPal Tel-Aviv. Product page classification is crucial for tasks like:
- E-commerce systems: Identifying product-related information for recommendation engines and pricing analysis.
- Web crawling: Filtering large-scale datasets to store only relevant product pages.
- Fraud detection: Ensuring product pages are legitimate and comply with regulations.

## Methodology
The classification task is broken down into three stages:

1. **URL-Based Classification:**
   - **Product Page Classification:** Identifying pages guaranteed to be product pages based on URL patterns (e.g., `/products/` or `/collections/.*/products/`).
   - **Non-Product Page Classification:** Excluding pages guaranteed not to be product pages (e.g., `/about-us`, `/contact-us`).

2. **Keyword-Based Classification:**
   - Analyzing the presence of common keywords (e.g., `add to cart`, `product description`) in the content of the web pages.
   - Pages with at least two characteristic keywords are classified as product pages.

3. **Embedding-Based Classification:**
   - Using SBERT (sentence-transformers/all-MiniLM-L6-v2) to embed web page content and compare it to labeled product page embeddings for semantic similarity.
   - Classifying based on a similarity threshold.

## Dataset
- Initial dataset: 100,000 homepage URLs provided by the mentors during the internship.
- Sub-links from these URLs were collected using the Scrapy framework.
- The project for collecting the dataset can be found in this repository: [PayPal Seed URLs Web Crawling EDA](https://github.com/raniakh/Paypal_Seed_URLs_WebCrawling_EDA).
- Challenges addressed during web crawling:
  - Circular loops and duplicate URLs.
  - Timeout errors.
  - Large data volume handled efficiently with Parquet files.

## Key Results
| Approach                | Accuracy | Precision | Recall |
|-------------------------|----------|-----------|--------|
| URL-Based Product Pages | 95.2%    | 90.5%     | 98.2%  |
| Keyword-Based           | 91.69%    | 80.0%     | 29.4%  |
| Embedding-Based         | 84.6%    | 7.25%     | 14.5%  |

## Code Structure
### 1. **`dataAnalysis.py`**
   - Includes exploratory data analysis (EDA) for URLs and content.

### 2. **`WebContentAnalysis.py`**
   - Contains utilities for analyzing web page content.

### 3. **`dataPrep.py`**
   - Prepares data for classification tasks (e.g., cleaning, processing).

### 4. **`product_classification_regex.py`**
   - Uses regex patterns to identify product and non-product pages from URLs.

### 5. **`product_classification_content.py`**
   - Implements content-based classification using keywords and embeddings.

### 6. **`product_classification_compare_webpage_embeddings.py`**
   - Compares embedding vectors to classify pages based on similarity scores.

## Challenges and Learnings
1. **Challenges:**
   - Dataset bias due to overrepresentation of Shopify-built pages.
   - Difficulty embedding long content while retaining product-specific signals.
   - Ineffectiveness of embeddings for sentence-level similarity.

2. **Key Learnings:**
   - URL patterns provide a quick and effective way to classify pages.
   - Simple keyword-based approaches can be surprisingly effective.
   - Embedding methods require better dataset quality and context-aware training.

## Future Improvements
- Fine-tune the SBERT model to improve embedding-based classification.
- Include non-HTML content types in the analysis.
- Perform deeper analysis to identify additional patterns.
- Balance the dataset to reduce Shopify bias.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/raniakh/PayPal_classifying_urls.git
   cd PayPal_classifying_urls
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run any of the scripts:
   ```bash
   python <script_name>.py
   ```
   Replace `<script_name>` with the desired script name (e.g., `dataAnalysis.py`).

## Thanks
I would like to express my gratitude to my mentors at PayPal Tel-Aviv for their invaluable guidance and support throughout this project. Their insights and expertise greatly contributed to its success.

## Contact
For questions or feedback, feel free to reach out:
- **Email:** raniakhoury07@gmail.com
- **LinkedIn:** [Rania Khoury](https://www.linkedin.com/in/raniakhoury7/)
- **Presentation:** [End of Internship Presentation](https://docs.google.com/presentation/d/1-Vyn9ng1n2JrCMoLP-VV7rMV-wr2oNq4ZQIcXM8p6rg/edit?usp=sharing)
