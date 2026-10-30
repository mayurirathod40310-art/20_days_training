
import csv
from pathlib import Path


def count_csv_rows(file_path: str) -> int:
    """Count the data rows in a CSV file, excluding the header."""

    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header row
        return sum(1 for row in reader if row)


def main() -> None:
    csv_file = Path(__file__).parent / "data" / "students.csv"

    try:
        total_rows = count_csv_rows(str(csv_file))
        print(f"CSV File: {csv_file.name}")
        print(f"Total Rows: {total_rows}")

    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")


if __name__ == "__main__":
    main()

