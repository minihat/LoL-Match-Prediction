from riotwatcher import RiotWatcher
import pprint
from operator import add
import numpy as np
import csv


watcher = RiotWatcher('RGAPI-c17a80bf-5625-4f65-8d20-c36d4e9d06e7')
my_region = 'na1'

# Open a file to write collected data to
csvfile = open('minimized_big_game_data_both_teams.csv','a')
bibowriter = csv.writer(csvfile,delimiter=',')


# Loop for as many matches as we can collect data for
matchidnumber = str(2645825000)
while True:
    try:
        # Get data from a match with id matchidnumber. We will attempt to predict
        #  the winner of this match
        ranked_game = 1
        match_data = watcher.match.by_id(my_region,matchidnumber)
        if int(match_data['queueId']) != 420:
            ranked_game = 0
            raise ValueError('Not a Ranked Game! Skipping to Next')
        #pprint.pprint(match_data)
        win_check = match_data['participants'][0]['stats']['win']
        if str(win_check) == 'True':
            win_check = 1
        else:
            win_check = 0

        # Get the players account ID's in a list (from one team)
        player_account_id_collection = []
        for playernumber in range(10):
            account_id = match_data['participantIdentities'][playernumber]['player']['accountId']
            player_account_id_collection.append(account_id)

        # Get the match history of each player
        player_matchlist_collection = []
        for player_account_id in player_account_id_collection:
            #matchlist = watcher.match.matchlist_by_account(my_region,player_account_id, queue=[400,420,430,440])
            matchlist = watcher.match.matchlist_by_account(my_region,player_account_id, queue=[420])
            just_the_match_ids = []
            for match in matchlist['matches']:
                just_the_match_ids.append(match['gameId'])
            # We only want the games that occurred prior to the game being analyzed
            largest_match_num = just_the_match_ids[0]
            while int(largest_match_num) >= int(matchidnumber):
                just_the_match_ids.pop(0)
                largest_match_num = just_the_match_ids[0]
            just_the_match_ids = just_the_match_ids[:10]
            player_matchlist_collection.append(just_the_match_ids)
        #pprint.pprint(player_matchlist_collection)



        # Get stats from the list of matches compiled above for team 1
        all_player_stats = [0] * 8
        for team_player_number in range(5):
            player_account_id_to_analyze = player_account_id_collection[team_player_number]

            num_successful_games = 0
            summed_player_stats = [0] * 8
            for match_to_analyze in player_matchlist_collection[team_player_number]:
                print("Analyzing game " + str(match_to_analyze) + " from player " + str(int(team_player_number)+1))
                match_data_2 = watcher.match.by_id(my_region,match_to_analyze)
                # Get the participant ID for our player account number
                for player_list in range(10):
                    if int(player_account_id_to_analyze) == int(match_data_2['participantIdentities'][player_list]['player']['accountId']):
                        participant_ID = match_data_2['participantIdentities'][player_list]['participantId']
                        #print(participant_ID)
                # Collect stats for our player from that game, and add them to the a list
                player_game_stats = match_data_2['participants'][int(participant_ID)-1]['stats']
                sorted_game_stats = []
                #keylist = list(player_game_stats.keys())
                keylist = ['assists','deaths','kills','totalDamageDealt','totalDamageDealtToChampions','totalMinionsKilled','visionScore','win']
                keylist.sort()
                for key in keylist:
                    sorted_game_stats.append(player_game_stats[key])
                final_stats_list = []
                for stat in sorted_game_stats:
                    statappend = stat
                    if str(stat) == 'True':
                        statappend = 1
                    if str(stat) == 'False':
                        statappend = 0
                    final_stats_list.append(statappend)
                # Get special stats from the game timeline data
                #timeline_data = match_data_2['participants'][int(participant_ID)-1]['timeline']
                #del timeline_data['lane']
                #del timeline_data['participantId']
                #del timeline_data['role']
                #timeline_useful_data = []
                #my_values = list(timeline_data.values())
                #for value in my_values:
            #        value_list = list(value.values())
        #            mean_value = sum(value_list) / float(len(value_list))
    #                timeline_useful_data.append(mean_value)
#                final_stats_list.extend(timeline_useful_data)
                #pprint.pprint(final_stats_list)
                #print(len(final_stats_list))
                if len(final_stats_list) == 8:
                    summed_player_stats = list(np.array(summed_player_stats) + np.array(final_stats_list))
                    num_successful_games += 1
            if num_successful_games > 0:
                summed_player_stats[:] = [float(x) / float(num_successful_games) for x in summed_player_stats]
            #all_player_stats.append(summed_player_stats)
            all_player_stats = list(np.array(all_player_stats) + np.array(summed_player_stats))
        #pprint.pprint(all_player_stats)
        #print("Win check: " + str(win_check))

        team1vector = all_player_stats

        # Get stats from the list of matches compiled for team 2
        all_player_stats = [0] * 8
        for team_player_number in range(5,10):
            player_account_id_to_analyze = player_account_id_collection[team_player_number]

            num_successful_games = 0
            summed_player_stats = [0] * 8
            for match_to_analyze in player_matchlist_collection[team_player_number]:
                print("Analyzing game " + str(match_to_analyze) + " from player " + str(int(team_player_number)+1))
                match_data_2 = watcher.match.by_id(my_region,match_to_analyze)
                # Get the participant ID for our player account number
                for player_list in range(10):
                    if int(player_account_id_to_analyze) == int(match_data_2['participantIdentities'][player_list]['player']['accountId']):
                        participant_ID = match_data_2['participantIdentities'][player_list]['participantId']
                        #print(participant_ID)
                # Collect stats for our player from that game, and add them to the a list
                player_game_stats = match_data_2['participants'][int(participant_ID)-1]['stats']
                sorted_game_stats = []
                #keylist = list(player_game_stats.keys())
                keylist = ['assists','deaths','kills','totalDamageDealt','totalDamageDealtToChampions','totalMinionsKilled','visionScore','win']
                keylist.sort()
                for key in keylist:
                    sorted_game_stats.append(player_game_stats[key])
                final_stats_list = []
                for stat in sorted_game_stats:
                    statappend = stat
                    if str(stat) == 'True':
                        statappend = 1
                    if str(stat) == 'False':
                        statappend = 0
                    final_stats_list.append(statappend)
                # Get special stats from the game timeline data
                #timeline_data = match_data_2['participants'][int(participant_ID)-1]['timeline']
                #del timeline_data['lane']
                #del timeline_data['participantId']
                #del timeline_data['role']
                #timeline_useful_data = []
                #my_values = list(timeline_data.values())
                #for value in my_values:
            #        value_list = list(value.values())
        #            mean_value = sum(value_list) / float(len(value_list))
    #                timeline_useful_data.append(mean_value)
#                final_stats_list.extend(timeline_useful_data)
                #pprint.pprint(final_stats_list)
                #print(len(final_stats_list))
                if len(final_stats_list) == 8:
                    summed_player_stats = list(np.array(summed_player_stats) + np.array(final_stats_list))
                    num_successful_games += 1
            if num_successful_games > 0:
                summed_player_stats[:] = [float(x) / float(num_successful_games) for x in summed_player_stats]
            #all_player_stats.append(summed_player_stats)
            all_player_stats = list(np.array(all_player_stats) + np.array(summed_player_stats))
        #pprint.pprint(all_player_stats)
        #print("Win check: " + str(win_check))

        team2vector = all_player_stats



        # Prepare the data line to write to csv file
        team1vector.extend(team2vector)
        write_vector = team1vector
        #for item in all_player_stats:
        #    write_vector.extend(item)
        write_vector.append(win_check)

        # Send data from this match to the csv file
        bibowriter.writerow(write_vector)
        csvfile.flush()
    except:
        if ranked_game == 0:
            print("Skipping due to: not a ranked game.")
        else:
            print("Skipping a game due to data retrieval failure.")
    matchidnumber = str(int(matchidnumber) + 1)
