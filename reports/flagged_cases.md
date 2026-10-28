# Reviewer case file: flagged applications

Model: `lightgbm` with TreeSHAP reason codes. Test window: months 6 and 7, 205,011 applications, 2,878 confirmed fraud.
The alert threshold is the score that lets through 5% of legitimate applications on the calibration window (month 5). On the test window it produced 14,094 alerts, of which 1,706 are fraud (12.1% precision). The 15 highest-scoring are listed below.

`Label` is shown only so the examples can be checked; a reviewer would not have it.

## Case 1: calibrated probability 0.90

- Month 6, customer age 50
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.05 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.84 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.72 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 2: calibrated probability 0.87

- Month 7, customer age 40
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.07 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.91 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.75 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 3: calibrated probability 0.86

- Month 7, customer age 70
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.02 log-odds)
  2. `credit_risk_score` = 338 (training percentile 100) raises the score (+0.92 log-odds)
  3. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.88 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 4: calibrated probability 0.85

- Month 7, customer age 30
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.01 log-odds)
  2. `device_os` = windows (27% of training rows) raises the score (+0.77 log-odds)
  3. `device_distinct_emails_8w` = 2 (training percentile 97) raises the score (+0.73 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 5: calibrated probability 0.84

- Month 6, customer age 60
- Label (demo only): legitimate
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.02 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.95 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.74 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 6: calibrated probability 0.83

- Month 7, customer age 60
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.06 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.95 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.70 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 7: calibrated probability 0.83

- Month 7, customer age 50
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.04 log-odds)
  2. `credit_risk_score` = 332 (training percentile 100) raises the score (+0.84 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.73 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 8: calibrated probability 0.82

- Month 6, customer age 70
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.14 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+0.89 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.75 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 9: calibrated probability 0.82

- Month 6, customer age 50
- Label (demo only): legitimate
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.13 log-odds)
  2. `credit_risk_score` = 321 (training percentile 99) raises the score (+0.81 log-odds)
  3. `date_of_birth_distinct_emails_4w` = 2 (training percentile 2) raises the score (+0.53 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 10: calibrated probability 0.80

- Month 7, customer age 30
- Label (demo only): legitimate
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.11 log-odds)
  2. `proposed_credit_limit` = 1900 (training percentile 99) raises the score (+1.01 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.81 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 11: calibrated probability 0.79

- Month 6, customer age 60
- Label (demo only): legitimate
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.09 log-odds)
  2. `device_os` = windows (27% of training rows) raises the score (+0.74 log-odds)
  3. `date_of_birth_distinct_emails_4w` = 1 (training percentile 0) raises the score (+0.55 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 12: calibrated probability 0.79

- Month 7, customer age 40
- Label (demo only): legitimate
- Why it was flagged:
  1. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+1.01 log-odds)
  2. `housing_status` = BA (16% of training rows) raises the score (+1.00 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.78 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 13: calibrated probability 0.79

- Month 7, customer age 80
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.07 log-odds)
  2. `proposed_credit_limit` = 1900 (training percentile 99) raises the score (+0.83 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.80 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 14: calibrated probability 0.79

- Month 6, customer age 60
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+0.99 log-odds)
  2. `credit_risk_score` = 316 (training percentile 99) raises the score (+0.86 log-odds)
  3. `device_os` = windows (27% of training rows) raises the score (+0.77 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason

## Case 15: calibrated probability 0.79

- Month 7, customer age 40
- Label (demo only): FRAUD
- Why it was flagged:
  1. `housing_status` = BA (16% of training rows) raises the score (+1.13 log-odds)
  2. `proposed_credit_limit` = 2000 (training percentile 99) raises the score (+1.03 log-odds)
  3. `credit_risk_score` = 314 (training percentile 99) raises the score (+0.86 log-odds)
- Review checklist:
  - [ ] Verify the applicant's identity documents against the declared date of birth and address
  - [ ] Check the email, phone and device against other recent applications (shared device, repeated email patterns)
  - [ ] Check the features named above for a legitimate explanation
  - [ ] Contact the applicant where policy requires it
  - [ ] Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason
