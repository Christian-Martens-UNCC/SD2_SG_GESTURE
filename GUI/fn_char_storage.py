def char_storage(char, hist_list, hist_len):
    """
    :param char: The character/input you want to add to the hist_list
    :param hist_list: The list of most recent inputs
    :param hist_len: The desired length of the history list. Longer lists are more accurate but also slow the model down
    :return: freq_char: The most popular character in the list
    :return: updated_list: The updated history list
    """

    updated_list = hist_list    # Store hist_list
    updated_list.insert(0, char)    # Insert the character into the list at the beginning
    while len(updated_list) > hist_len:     # Checks if the list is now longer than it should be
        updated_list = updated_list[:-1]    # Removes the end of the list until the desired size is reached
    freq_char = max(set(updated_list), key=updated_list.count)  # Finds the most common character
    return freq_char, updated_list
