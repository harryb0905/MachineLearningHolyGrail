import pandas as pd
import csv

def read_file_as_df(filepath, delimiter=','):
  return pd.read_csv(filepath, delimiter=delimiter)

def write_df(df, filepath, delimiter=','):
  df.to_csv(filepath, index=False)

def main():
  filepath = '../datasets/practice/people.csv'
  people_df = read_file_as_df(filepath)
  print(people_df.head())

  # Write list of rows to CSV
  rows = [
    ["9000", "24.99"],
    ["9250", "32.99"]
  ]
  columns = ["ItemId", "Price"]
  items_df = pd.DataFrame(rows, columns=columns)

  filepath = 'items.csv'
  # result = write_df(items_df, filepath, rows)

  # Write list of dicts to CSV
  rows = [
    {'player_name': 'Magnus Carlsen', 'fide_rating': 2870},
    {'player_name': 'Fabiano Caruana', 'fide_rating': 2822},
    {'player_name': 'Ding Liren', 'fide_rating': 2801}
  ]
  columns = ['player_name', 'fide_rating']
  players_df = pd.DataFrame(rows, columns=columns)

  filepath = 'players.csv'
  # result = write_df(players_df, filepath, rows)

if __name__ == "__main__":
  main()
