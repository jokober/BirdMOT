def improve_text_position(x):
    """ it is more efficient if the x values are sorted """
    # fix indentation
    positions = ['top center', 'bottom center']  # you can add more: left center ...
    return [positions[i % len(positions)] for i in range(len(x))]