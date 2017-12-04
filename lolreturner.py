from riotwatcher import RiotWatcher
import pprint
import sys
import csv

# Get match id from command line
#matchidnumber = sys.argv[1]
dumb_predict_accuracy = 0
accurate_predictions = 0
total_predictions = 0

watcher = RiotWatcher('RGAPI-d4858353-dad7-4f41-ac51-dd8d7a21d812')

my_region = 'na1'

csvfile = open('gameslist3.csv','a')
bibowriter = csv.writer(csvfile,delimiter=',')


#me = watcher.summoner.by_name(my_region, 'Lahourla')
#print(me)
#player_id = me['id']
#player_account_id = me['accountId']
#champ_list = watcher.champion.all(my_region)
#print(champ_list)

#player_champ_masteries = watcher.champion_mastery.by_summoner_by_champion(my_region,player_id,'154')
#print(player_champ_masteries)
#champion_points = player_champ_masteries['championPoints']
#print(champion_points)

#matchlist = watcher.match.matchlist_by_account(my_region,player_account_id)
#print(matchlist)


#Automatically cycle through some recently played games
corrector = 0
for game in range(100000):
    matchidnumber = str(2645825000+game+corrector)
    while True:
        try:
            # Get match data for the current match to analyze
            match_data = watcher.match.by_id(my_region,matchidnumber)
            break
            #pprint.pprint(match_data)
        except:
            corrector += 1
            matchidnumber = str(int(matchidnumber)+1)

    game_mastery_list = []
    for player in range(10):
        try:
            # Get the summoner identification number
            summ_id = match_data['participantIdentities'][player]['player']['summonerId']
            #print(summ_id)
        except:
            summ_id = 0

        try:
            # Get the champion ID for the champ this player is using
            champ_id = match_data['participants'][player]['championId']
            #print(champ_id)
        except:
            champ_id = 0
        # Lookup how experienced this player is on the above champ
        try:
            player_champ_mastery_info = watcher.champion_mastery.by_summoner_by_champion(my_region,summ_id,champ_id)
            champ_mastery = player_champ_mastery_info['championPoints']
        except:
            champ_mastery = 0
        #print(champ_mastery)
        game_mastery_list.append(champ_mastery)
        print("Player " + str(player+1) + " has summoner ID: " + str(summ_id) + " champion ID: " + str(champ_id) + " champion mastery: " + str(champ_mastery))

    # Find out who won the game
    win = 0
    team1win = match_data['participants'][0]['stats']['win']
    print(team1win)
    if team1win:
        win = 1

    print("Team 1 summed mastery: " + str(sum(game_mastery_list[:5])))
    print("Team 2 summed mastery: " + str(sum(game_mastery_list[5:])))
    print("Did team 1 win? " + str(win))
    game_mastery_list.append(str(win))
    bibowriter.writerow(game_mastery_list)
    csvfile.flush()
    t1 = sum(game_mastery_list[:5])
    t2 = sum(game_mastery_list[5:-1])
    if t1 > t2 and win == 1:
        accurate_predictions += 1
    if t1 < t2 and win == 0:
        accurate_predictions += 1
    total_predictions += 1
    dumb_predict_accuracy = 100 * float(accurate_predictions)/float(total_predictions)
    print("Game number " + str(total_predictions) + " analyzed.")
    print("Dumb Prediction Accuracy this Far: " + str(dumb_predict_accuracy) + "%")

csvfile.close()
