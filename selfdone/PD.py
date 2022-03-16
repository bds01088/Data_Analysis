import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic/train.csv")

print (df.info())

alive = df[df["Survived"] == 1]
dead = df[df["Survived"] == 0]

# print (len(df))
# print (len(alive))
# print (len(dead))

# plt.bar(["alive", "dead"], height = [len(alive), len(dead)])
# plt.show()

# plt.scatter(alive["PassengerId"], alive["Fare"], color = "GREEN")
# plt.scatter(dead["PassengerId"], dead["Fare"], color = "RED")

# plt.xlabel("PassengerId")
# plt.ylabel("Fare")
# plt.show()

h = df[df["Fare"] >= 50]
l = df[df["Fare"] < 50]

# print (round(len(h)/len(df)*100))
# print (round(len(l)/len(df)*100))

hl = h[h["Survived"] == 1]
hd = h[h["Survived"] == 0]
ll = l[l["Survived"] == 1]
ld = l[l["Survived"] == 0]

# print (len(hl), len(hd))
# print (len(ll), len(ld))

plt.subplot(2,1,1)
plt.pie([len(hl), len(hd)])
plt.subplot(2,1,2)
plt.pie([len(ll), len(ld)])
plt.show()