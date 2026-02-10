# Data Dictionary

- Generated: 2026-02-10 14:41:24
- Horizon (months): 12
- Variables: 20

## 1. age_years

- Type: `numeric`
- Expected range: `18` to `120`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `18-24` | 9.0 | 2.1667 |
| `25-34` | 21.0 | 1.6 |
| `35-44` | 26.0 | 1.1333 |
| `45-54` | 22.0 | 1.0 |
| `55-64` | 16.0 | 1.1667 |
| `65+` | 6.0 | 1.7333 |

## 2. residential_status

- Type: `categorical`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `owner_no_mortgage` | 18.0 | 1.0 |
| `owner_with_mortgage` | 32.0 | 1.2727 |
| `rent_private` | 30.0 | 1.61 |
| `rent_social` | 12.0 | 2.1 |
| `living_with_family_other` | 8.0 | 1.8 |

## 3. time_at_address_months

- Type: `numeric`
- Expected range: `0` to `600`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<6` | 10.0 | 2.2 |
| `6-12` | 15.0 | 2.0714 |
| `12-24` | 20.0 | 1.6429 |
| `24-60` | 30.0 | 1.25 |
| `>60` | 25.0 | 1.0 |

## 4. employment_status

- Type: `categorical`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `permanent` | 62.0 | 1.0 |
| `temporary_contract` | 14.0 | 1.7419 |
| `self_employed` | 12.0 | 1.3871 |
| `retired` | 7.0 | 1.1613 |
| `unemployed_other` | 5.0 | 2.3 |

## 5. time_with_employer_months

- Type: `numeric`
- Expected range: `0` to `600`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<6` | 12.0 | 2.3517 |
| `6-12` | 16.0 | 2.0345 |
| `12-24` | 18.0 | 1.5862 |
| `24-60` | 28.0 | 1.2069 |
| `>60` | 26.0 | 1.0 |

## 6. gross_monthly_income_eur

- Type: `numeric`
- Expected range: `0` to `20000`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<1500` | 12.0 | 2.1 |
| `1500-2500` | 30.0 | 1.8 |
| `2500-3500` | 28.0 | 1.5 |
| `3500-5000` | 20.0 | 1.35 |
| `>5000` | 10.0 | 1.0 |

## 7. net_monthly_income_eur

- Type: `numeric`
- Expected range: `0` to `15000`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<1200` | 15.0 | 2.2 |
| `1200-1800` | 32.0 | 1.9 |
| `1800-2500` | 30.0 | 1.65 |
| `2500-3500` | 18.0 | 1.3 |
| `>3500` | 5.0 | 1.0 |

## 8. debt_to_income_ratio

- Type: `numeric`
- Expected range: `0` to `1.5`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<0.20` | 18.0 | 1.0 |
| `0.20-0.35` | 30.0 | 1.3636 |
| `0.35-0.50` | 28.0 | 1.9545 |
| `0.50-0.65` | 16.0 | 2.25 |
| `>0.65` | 8.0 | 2.8 |

## 9. disposable_income_eur

- Type: `numeric`
- Expected range: `-5000` to `20000`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<0` | 6.0 | 2.9 |
| `0-300` | 18.0 | 1.8 |
| `300-700` | 32.0 | 1.5 |
| `700-1200` | 28.0 | 1.3333 |
| `>1200` | 16.0 | 1.0 |

## 10. num_active_credit_accounts

- Type: `numeric`
- Expected range: `0` to `50`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `0-1` | 14.0 | 1.7059 |
| `2-3` | 34.0 | 1.0588 |
| `4-5` | 28.0 | 1.0 |
| `6-8` | 18.0 | 1.3529 |
| `>8` | 6.0 | 2.0294 |

## 11. total_outstanding_balance_eur

- Type: `numeric`
- Expected range: `0` to `200000`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<2000` | 18.0 | 1.0 |
| `2000-5000` | 24.0 | 1.1613 |
| `5000-10000` | 26.0 | 1.3871 |
| `10000-20000` | 20.0 | 1.7419 |
| `>20000` | 12.0 | 2.3226 |

## 12. credit_utilisation_ratio

- Type: `numeric`
- Expected range: `0` to `1.5`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<0.20` | 22.0 | 1.0 |
| `0.20-0.40` | 28.0 | 1.3333 |
| `0.40-0.60` | 24.0 | 1.8333 |
| `0.60-0.80` | 16.0 | 2.333 |
| `>0.80` | 10.0 | 2.888 |

## 13. num_unsecured_loans

- Type: `numeric`
- Expected range: `0` to `20`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `0` | 18.0 | 1.0 |
| `1` | 36.0 | 1.1333 |
| `2` | 26.0 | 1.5 |
| `3` | 14.0 | 2.0333 |
| `>=4` | 6.0 | 2.4 |

## 14. num_accounts_30dpd_last12m

- Type: `numeric`
- Expected range: `0` to `20`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `0` | 76.0 | 1.0 |
| `1` | 14.0 | 1.5 |
| `2` | 6.0 | 2.1 |
| `>=3` | 4.0 | 2.9 |

## 15. max_delinquency_last12m

- Type: `ordinal`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `current` | 78.0 | 1.0 |
| `30dpd` | 12.0 | 1.4 |
| `60dpd` | 6.0 | 1.9 |
| `90dpd_plus` | 4.0 | 2.3 |

## 16. months_since_last_delinquency

- Type: `numeric`
- Expected range: `0` to `240`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `never` | 58.0 | 1.0 |
| `<6` | 10.0 | 3.2 |
| `6-12` | 12.0 | 2.8 |
| `12-24` | 10.0 | 1.8 |
| `>24` | 10.0 | 1.41 |

## 17. worst_ever_delinquency

- Type: `ordinal`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `none` | 62.0 | 1.0 |
| `30dpd` | 16.0 | 2.1154 |
| `60dpd` | 10.0 | 3.2692 |
| `90dpd_plus` | 12.0 | 5.1923 |

## 18. credit_history_length_months

- Type: `numeric`
- Expected range: `0` to `600`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `<12` | 8.0 | 2.4286 |
| `12-36` | 22.0 | 1.75 |
| `36-72` | 28.0 | 1.3214 |
| `72-120` | 26.0 | 1.1071 |
| `>120` | 16.0 | 1.0 |

## 19. num_credit_enquiries_last6m

- Type: `numeric`
- Expected range: `0` to `20`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `0` | 42.0 | 1.0 |
| `1` | 28.0 | 1.2857 |
| `2` | 16.0 | 1.7143 |
| `3` | 8.0 | 2.4286 |
| `>=4` | 6.0 | 3.5 |

## 20. adverse_events_flag

- Type: `binary`

| Band | Distribution (%) | Bad Rate Ratio |
|---|---:|---:|
| `no` | 86.0 | 1.0 |
| `yes` | 14.0 | 2.2 |

