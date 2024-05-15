### Utilities to get overlap between strings

def get_overlap_length(left: str, right: str):
    good_length, overlap = 0, ""
    for i in range(min(len(left), len(right))):
        if left[-i:] == right[:i]:
            good_length = i
            overlap = left[-i:]
    return good_length, overlap

def get_overlap_list(strings):
    """
    Returns a list of tuples of the form (overlap_length, overlap), one tuple for each pair of strings in the input list.
    """
    overlaps = []
    for i in range(len(strings) - 1):
        overlaps.append(get_overlap_length(strings[i], strings[i+1]))
    return overlaps

def unoverlap_list(strings):
    """
    Returns a list of tuples of the form (content, is_overlap), where is_overlap is a boolean indicating whether the content is an overlap or not.
    """
    overlaps = get_overlap_list(strings)
    new_list = []
    for index, string in enumerate(strings):
        # Add the last overlap when needed
        if index > 0 and len(overlaps[index-1][1]) > 0:
            new_list.append((overlaps[index-1][1], True))

        # prune the string with left and right overlaps
        left_overlap_length, right_overlap_length = 0, 0
        if index > 0:
            left_overlap_length = overlaps[index-1][0]
        if index < len(strings) - 1:
            right_overlap_length = overlaps[index][0]

        new_list.append((string[left_overlap_length:len(string)-right_overlap_length], False))
    return new_list