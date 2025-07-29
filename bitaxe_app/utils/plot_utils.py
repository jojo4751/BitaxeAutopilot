def make_plotly_traces(rows):
    """
    Erzeugt Plotly-Traces aus den DB-Rows der Tabelle efficiency_markers.
    rows: Liste von Tuples in der Reihenfolge:
      (timestamp, ip, hostname, temp, hashRate, power,
       frequency, coreVoltage, sharesAccepted, sharesRejected,
       watt_per_share, gh_per_watt, shares_per_gh, uptime,
       shares_accepted_delta, shares_rejected_delta,
       shares_per_hour, gh_per_share)
    """
    # Zeitachse
    timestamps = [row[0] for row in rows]

    # Spalten extrahieren
    temp                  = [row[3]  for row in rows]
    hash_rate             = [row[4]  for row in rows]
    power                 = [row[5]  for row in rows]
    frequency             = [row[6]  for row in rows]
    core_voltage          = [row[7]  for row in rows]
    shares_accepted       = [row[8]  for row in rows]
    shares_rejected       = [row[9]  for row in rows]
    watt_per_share        = [row[10] for row in rows]
    gh_per_watt           = [row[11] for row in rows]
    shares_per_gh         = [row[12] for row in rows]
    uptime                = [row[13] for row in rows]
    shares_acc_delta      = [row[14] for row in rows]
    shares_rej_delta      = [row[15] for row in rows]
    shares_per_hour       = [row[16] for row in rows]
    gh_per_share          = [row[17] for row in rows]

    # Traces definieren
    traces = [
        {'x': timestamps, 'y': temp,            'name': 'Temperatur',     'mode': 'lines'},
        {'x': timestamps, 'y': hash_rate,       'name': 'HashRate',       'mode': 'lines'},
        {'x': timestamps, 'y': power,           'name': 'Power (W)',      'mode': 'lines'},
        {'x': timestamps, 'y': watt_per_share,  'name': 'Watt/Share',     'mode': 'lines'},
        {'x': timestamps, 'y': gh_per_watt,     'name': 'GH/W',           'mode': 'lines'},
        {'x': timestamps, 'y': gh_per_share,    'name': 'GH/Share',       'mode': 'lines'},
        # Bei Bedarf weitere Traces wie frequency, core_voltage, uptime, etc.
    ]

    return traces
