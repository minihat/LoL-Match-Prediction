import csv
import numpy

csvfile = open('gameslist3.csv','r')
reader = csv.reader(csvfile)

team1wins = 0
for row in reader:
    try:
        win = row[-1]
    except:
        win = 0
    team1wins += int(win)

print("Total number of wins: " + str(team1wins))
