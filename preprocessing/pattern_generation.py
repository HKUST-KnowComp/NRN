import csv


def get_patterns():
    all_patterns = []
    with open('test_generated_formula_anchor_node=3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(f'{row[1]} .')
                all_patterns.append(row[1])
                line_count += 1
        print(f'Processed {line_count} lines.')
    return all_patterns


if __name__ == "__main__":

    print(get_patterns())

