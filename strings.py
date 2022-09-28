def unpack_tuple(tup):
    returned = ""
    for item in tup:
        returned += "," + str(item)
    return returned[1:]