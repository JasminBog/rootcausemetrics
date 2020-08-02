path = "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/novel_data/log.txt"
pom = []
with open(path, "r") as file:
    content = file.readlines()
    for _, x in enumerate(content):
        print(x)
        if "Service:" in x:
            pom.append(x)
        elif "Fault start-point:" in x:
            pom.append(x)
        else:
            continue
path_to_store = path[:-4] + "_filtered.txt"
with open(path_to_store, "w") as file1:
    file1.writelines(pom)