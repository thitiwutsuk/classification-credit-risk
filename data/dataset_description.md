# Dataset Description

## Context

Customer transaction and demographic data — holds Risky and Not Risky customers for a specific banking product.

Source: Google Datasets Search.

## Files

### `payment_data.csv`

Customer's card payment history.

| Column | Description |
|---|---|
| `id` | Customer id |
| `OVD_t1` | Number of times overdue, type 1 |
| `OVD_t2` | Number of times overdue, type 2 |
| `OVD_t3` | Number of times overdue, type 3 |
| `OVD_sum` | Total overdue days |
| `pay_normal` | Number of times paid normally |
| `prod_code` | Credit product code |
| `prod_limit` | Credit limit of the product |
| `update_date` | Account update date |
| `new_balance` | Current balance of the product |
| `highest_balance` | Highest balance in history |
| `report_date` | Date of the most recent payment |

### `customer_data.csv`

Customer's demographic data and encoded category attributes.

- `fea_1, fea_3, fea_5, fea_6, fea_7, fea_9` — categorical features (encoded)
- `label` — `1` = high credit risk, `0` = low credit risk

## Question this dataset is meant to answer

This dataset helps determine whether a customer is Credit Risky or Credit Worthy from a banking perspective.

- **Q1** — What are the factors contributing to a Credit Risky customer?
- **Q2** — What is the behavior of a Credit Worthy customer?
