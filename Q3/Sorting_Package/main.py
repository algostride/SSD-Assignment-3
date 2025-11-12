import sys
from src.sort_factory import SortFactory

def read_input(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [int(x.strip()) for x in lines if x.strip().isdigit()]

def main():
    if len(sys.argv) < 4:
        print("Usage: python main.py <algorithm> <asc/desc> <input_file>")
        sys.exit(1)

    algorithm = sys.argv[1].lower()
    ascending = sys.argv[2].lower() == "asc"
    input_file = sys.argv[3]

    data = read_input(input_file)
    sorter = SortFactory()
    sorted_list = sorter.sort(ascending, algorithm, len(data), data)

    print(f"\nAlgorithm: {algorithm.capitalize()}")
    print(f"Order: {'Ascending' if ascending else 'Descending'}")
    print(f"Input: {data}")
    print(f"Sorted Output: {sorted_list}")

if __name__ == "__main__":
    main()