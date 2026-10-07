# MetaQuotes-Demo EURUSD session calendar

The versioned runtime calendar is `config/broker_sessions/metaquotes-demo-eurusd.json`.
It encodes the observed regular FX week in `America/New_York` time so the UTC
open/close shifts with daylight-saving rules. It is intentionally not an
always-open calendar. The parser also supports full-date holidays and
date-specific special-session overrides; none are populated because no
authoritative broker exception schedule was available for this configuration.

The weekly profile was checked against normalized MT5 historical quote
boundaries for EURUSD on Sundays 2026-09-13, 2026-09-20, 2026-09-27, and
2026-10-04 (first ticks near 21:00 UTC, 17:00 EDT), and Fridays
2026-09-18, 2026-09-25, and 2026-10-02 (last ticks near 20:59:40-20:59:57
UTC, before 17:00 EDT). The `America/New_York` timezone handles the seasonal
UTC shift. Session entries end at 23:59 rather than using an unsupported
24:00 value; this leaves a conservative one-minute daily closed interval.

The calendar is only a lifecycle hint. A fresh broker tick still triggers
opening validation, and stale/missing quotes never become trade permission.
If observed broker behavior changes, update this file only from new broker
evidence and add the corresponding date override when the exception is
authoritatively known.
