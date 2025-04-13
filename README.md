# invoiceautomation
# Invoice Automation System

## Overview
This Python script implements an advanced invoice automation system that leverages Optical Character Recognition (OCR) to extract and classify data from scanned invoices, supplemented with generated data to simulate business-scale processing. Designed to streamline accounts payable workflows, it processes invoices to extract fields like invoice number, date, total amount, and vendor name, validates the data, and outputs a CSV file for integration with Power BI or other reporting tools. The system includes 50 simulated invoices to demonstrate scalability, ensuring robust output for professional evaluation.

## Features
- **OCR Processing**: Utilizes Tesseract OCR with OpenCV for accurate text extraction from invoice images.
- **Enhanced Preprocessing**: Applies perspective correction, thresholding, and noise reduction to optimize OCR performance.
- **Data Extraction**: Employs regular expressions for reliable field classification, with fallback for missing data.
- **Data Validation**: Uses Pydantic to enforce strict data integrity (e.g., valid dates, positive amounts).
- **Data Generation**: Generates 50 realistic invoices to ensure a populated output, simulating business data volume.
- **Cloud Integration**: Syncs output to Google Drive for persistent storage.
- **Performance**: Supports parallel processing for efficiency using ThreadPoolExecutor.
- **Observability**: Implements structured logging and basic metrics tracking for execution monitoring.
- **Security**: Includes basic PII redaction for sensitive data like email addresses.
- **Output**: Produces a Power BI-compatible CSV file, downloadable from Google Colab.

## Prerequisites
- **Environment**: Google Colab (cloud-based Python notebook).
- **Dependencies**: Automatically installed (`pytesseract`, `opencv-python`, `pandas`, `faker`, `pydantic`, `tesseract-ocr`).
- **Internet**: Required for library installation and Google Drive access.
- **Google Account**: Needed for Drive sync (optional).

## Usage
1. Copy and execute the script in a Google Colab notebook.
2. The pipeline will:
   - Generate a sample invoice image for OCR demonstration.
   - Extract data from the image using OCR and combine it with 50 simulated invoices.
   - Validate all data for integrity.
   - Save the results to `invoices_processed.csv`, which downloads automatically.
   - Sync the CSV to Google Drive (if authenticated).
3. Import the CSV into Power BI to create visualizations (e.g., total invoice amounts by vendor, date trends).
4. Download `invoice_pipeline.log` from Colab for execution details or troubleshooting.

## Data Sources
- **Sample Invoice Image**: A programmatically generated image with fields like vendor name, invoice number, date, and total amount for OCR testing.
- **Simulated Invoices**: 50 generated records with realistic data (invoice numbers INV-1001 to INV-1050, varied dates, amounts between $100 and $1000, and vendor names).
- **Real Invoices**: Supports uploaded images via `files.upload()` (update `image_path` in `run_invoice_pipeline`).

## Output
The pipeline produces `invoices_processed.csv` with columns:
- `invoice_number`: Unique identifier (e.g., INV-12345).
- `invoice_date`: Issuance date (e.g., 04/10/2025).
- `total_amount`: Total due (e.g., 125.00).
- `vendor_name`: Supplier name (e.g., Acme Corp).
- `timestamp`: Processing timestamp.

The CSV includes at least 50 invoices, ensuring a populated output for reporting.

## Technical Details
- **Preprocessing**: Enhances images with grayscale conversion, 2x resizing, perspective correction, Otsu’s thresholding, and Gaussian blur for OCR accuracy.
- **OCR Engine**: Uses Tesseract with `--oem 3 --psm 6` configuration for reliable text extraction.
- **Data Extraction**: Combines regex-based field detection with fallback values to handle OCR failures.
- **Validation**: Enforces data quality with Pydantic, ensuring valid dates and positive amounts.
- **Data Generation**: Leverages Faker for realistic invoice data, validated before inclusion.
- **Cloud Sync**: Saves output to Google Drive for persistence, with error handling for connectivity issues.
- **Performance**: Parallelizes tasks using ThreadPoolExecutor, optimized for Colab’s environment.
- **Security**: Redacts basic PII (e.g., emails) using regex to protect sensitive data.
- **Monitoring**: Tracks processed invoices and OCR errors, logged for transparency.

## Customization
To adapt for production:
- **Real Invoices**: Upload images via `files.upload()` and pass to `run_invoice_pipeline`.
- **Fields**: Extend regex patterns in `extract_invoice_data` for additional fields (e.g., tax, line items).
- **Validation**: Update the `Invoice` model for specific business rules.
- **Output**: Modify `save_to_csv` for database integration (e.g., Google BigQuery).
- **Preprocessing**: Adjust perspective correction or add deskewing for varied invoice layouts.
- **Deployment**: Deploy to a cloud platform (e.g., Google Cloud Functions) for automated processing.

## Limitations
- **Colab Environment**: Outputs (`invoices_processed.csv`, `invoice_pipeline.log`) are temporary unless synced to Google Drive.
- **OCR Accuracy**: Tesseract may struggle with complex layouts or low-quality scans; simulated data ensures output reliability.
- **Sample Data**: Simulated invoices supplement OCR for demonstration; real invoices may require tuned regex.
- **Google Drive**: Requires manual authentication in Colab for sync.

## Future Enhancements
- Integrate advanced OCR services (e.g., Google Cloud Vision) for improved accuracy.
- Support table extraction for line-item details using ML models.
- Add workflow orchestration (e.g., Apache Airflow) for scheduled runs in a production environment.
- Implement real-time alerting via cloud monitoring tools for error notifications.

## Notes
This system demonstrates proficiency in OCR, data validation, automation, and cloud integration, tailored for business applications like accounts payable. Its design emphasizes scalability, reliability, and clarity, making it ideal for showcasing data engineering skills to prospective employers.

For further information, please contact me on linkedin at https://www.linkedin.com/in/edward-antwi-8a01a1196/
