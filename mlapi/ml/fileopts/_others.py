import csv

PARSE_TYPES = { 
  'dict': csv.DictReader,
  'list': csv.reader,
  'df': pd.DataFrame
}

def read_csv_custom(filepath, delimiter=',', parse_type='dict'):
  with open(filepath, 'r') as file:
    if parse_type in PARSE_TYPES:
      reader = PARSE_TYPES[parse_type](file, delimiter=delimiter)
      for row in reader:
        print(row)

def write_csv(filepath, rows, headers=None, delimiter=',', parse_type='dict'):
  if len(rows):
    with open(filepath, 'w') as file:
      if parse_type in PARSE_TYPES:
        if parse_type == 'dict':
          if headers is None:
            # Extract headers using keys of first row (if not provided explicitly)
            headers = rows[0].keys()

          writer = csv.DictWriter(file, fieldnames=headers)
          writer.writeheader()
          writer.writerows(rows)
        elif parse_type == 'list':
          if headers is None:
            # Extract headers using values of first row (if not provided explicitly)
            headers = rows[0]

          writer = csv.writer(file, delimiter=delimiter)
          writer.writerow(headers)
          writer.writerows(rows)

        num_rows = len(rows)
        return f"Wrote {num_rows if headers is None else num_rows + 1} row(s) to {filepath}"
      else:
        return f"Unsupported parse type - rows should be a list of lists or a list of dicts. Should be one of: {', '.join(PARSE_TYPES.keys())}"
