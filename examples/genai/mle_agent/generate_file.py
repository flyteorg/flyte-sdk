with open("data.csv", "w") as f:
    f.write("feature1,feature2,target\n")
    for i in range(10_000):
        f.write(f"{i},{i * 2},{i * 3}\n")
