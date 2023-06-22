import re

def sort_by_date(filenames):
    """Sorts a list of tar filenames by date"""
    # Use a regular expression to parse the dates
    pattern = re.compile(r'arXiv_src_(\d{2})(\d{2})_(\d{3})\.tar')

    def get_date(filename):
        """Extracts a date tuple from a filename"""
        match = pattern.match(filename)
        if match is None:
            return 0, 0, 0  # if the filename does not match, return a dummy value
        year, month, number = match.groups()
        year = int(year)
        if year < 80:  # if the year is less than 80, it is in the 2000s
            year += 2000
        else:  # if the year is 21 or more, it is 1900s
            year += 1900
        return year, int(month), int(number)

    return sorted(filenames, key=get_date)

# usage
filenames = ["arXiv_src_9911_075.tar", "arXiv_src_2023_075.tar", "arXiv_src_2010_075.tar", "arXiv_src_2012_075.tar", "arXiv_src_9811_075.tar"]
sorted_filenames = sort_by_date(filenames)
print(sorted_filenames)
