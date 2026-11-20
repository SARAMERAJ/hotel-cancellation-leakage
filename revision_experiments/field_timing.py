"""Time-dependence check for every input field (Section IV-H).
Uses reservation_status_date for diagnosis only, never as a feature.
Check 1: kept bookings vs bookings cancelled 0-1 days before arrival.
Check 2: cancelled bookings with lead time 91-180 days, first vs last quartile of days between booking and cancellation;
         standardised mean difference (numeric) or total variation distance (categorical).
Fields fixed at booking (country, market segment) give the baseline difference caused by who cancels early."""
import numpy as np, pandas as pd
DATA, OUT = 'hotel_bookings.csv', 'out'
raw = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
raw = raw[(raw.adults + raw.children.fillna(0) + raw.babies) > 0].reset_index(drop=True)
raw['children'] = raw.children.fillna(0)
arr = pd.to_datetime(raw.arrival_date_year.astype(str) + '-' + raw.arrival_date_month + '-' +
                     raw.arrival_date_day_of_month.astype(str), format='%Y-%B-%d')
bk = arr - pd.to_timedelta(raw.lead_time, unit='D')
status = pd.to_datetime(raw.reservation_status_date)
c = raw.is_canceled == 1
num = ['stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest',
       'previous_cancellations', 'previous_bookings_not_canceled', 'days_in_waiting_list', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests']
cat = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type',
       'customer_type']
lm = c & (arr - status).dt.days.between(0, 1)
cc = raw[c].copy(); cc['days_in_system'] = (status - bk).dt.days[c]
band = cc[cc.lead_time.between(91, 180)]
qt = pd.qcut(band.days_in_system, 4).cat.codes
first, last = band[qt == 0], band[qt == 3]
rows = []
for f in num:
    s = np.sqrt((first[f].var() + last[f].var()) / 2)
    rows.append({'Field': f, 'Type': 'numeric', 'Kept: share > 0': (raw.loc[~c, f] > 0).mean(),
                 'Cancelled 0-1 days before arrival: share > 0': (raw.loc[lm, f] > 0).mean(),
                 'Early vs late cancellation (lead 91-180)': (last[f].mean() - first[f].mean()) / s if s > 0 else 0,
                 'Measure': 'standardised mean difference'})
for f in cat:
    p, r = first[f].value_counts(normalize=True), last[f].value_counts(normalize=True)
    rows.append({'Field': f, 'Type': 'categorical',
                 'Early vs late cancellation (lead 91-180)': 0.5 * p.subtract(r, fill_value=0).abs().sum(),
                 'Measure': 'total variation distance'})
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/field_timing.csv', index=False)
print(df.round(3).to_string())
