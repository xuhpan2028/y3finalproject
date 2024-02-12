class Card:

    def __init__(self, Number, Colour):
        self.__Number = Number
        self.__Colour = Colour
    
    def getNumber(self):
        return self.__Number
    
    def getColour(self):
        return self.__Colour
    

CardArray = [0]*30

try: 
    Filename = "CardValues.txt" 
    File = open(Filename,'r') 
    for x in range(30): 
        NumberRead = int(File.readline()) 
        ColourRead = File.readline() 
        CardArray[x] = Card(NumberRead, ColourRead) 
    File.close()
except IOError: 
    print("Could not find file") 

list = [True]*30
def ChooseCard():
    num = int(input("Enter the number: "))
    while num < 1 or num > 30 or list[num-1] == False:
        print("Invalid number")
        num = int(input("Enter the number: "))
    
    list[num-1] = False
    return num-1


Player1 = []
for i in range(4):
    ints = ChooseCard()
    Player1.append(CardArray[ints])

for i in range(4):
    print(Player1[i].getNumber(), Player1[i].getColour())





