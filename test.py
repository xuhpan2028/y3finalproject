class Character:
    
    def __init__(self, Name, XCoordinate, YCoordinate):
        self.__Name = Name                  
        self.__XCoordinate = XCoordinate    
        self.__YCoordinate = YCoordinate    

    
    def GetName(self):
        return self.__Name
    
    def GetX(self):
        return self.__XCoordinate
    
    def GetY(self):
        return self.__YCoordinate
    
    def ChangePosition(self, XChange, YChange):
        self.__XCoordinate += XChange
        self.__YCoordinate += YChange


Characters = [0]*10
try:
    Filename = "Characters.txt"
    File = open(Filename, "r")
    for i in range(10):
        Name = File.readline()
        X = int(File.readline())
        Y = int(File.readline())
        Characters[i] = Character(Name, X, Y)
    File.close()
except:
    print("can't open file")



Name = input("Enter the name of the character: ")
Flag = False
while Flag == False:
    for i in range(10):
        if Characters[i].GetName() == Name:
            Flag = True
            index = i
            break
    Name = input("Enter the name of the character: ")

Direction = input("Enter the direction (A, W, S, D): ")
while Direction != "A" and Direction != "W" and Direction != "S" and Direction != "D":
    Direction = input("Enter the direction (A, W, S, D): ")

if Direction == "A":
    Characters[index].ChangePosition(-1, 0)
elif Direction == "W":
    Characters[index].ChangePosition(0, 1)
elif Direction == "S":
    Characters[index].ChangePosition(0, -1)
elif Direction == "D":
    Characters[index].ChangePosition(1, 0)


print(Characters[index].GetName(), "has changed coordinates to X =", Characters[index].GetX(), "and Y =", Characters[index].GetY())