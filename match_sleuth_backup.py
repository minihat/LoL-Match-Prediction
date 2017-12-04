from riotwatcher import RiotWatcher
import pprint

watcher = RiotWatcher('RGAPI-683afda5-c62a-4cdd-bf76-7fdaa5a1223d')
my_region = 'na1'

# Get data from a match with id matchidnumber. We will attempt to predict
#  the winner of this match using data science
matchidnumber = str(2645825000)
match_data = watcher.match.by_id(my_region,matchidnumber)
#pprint.pprint(match_data)


# Get the players account ID's in a list (from one team)
player_account_id_collection = []
for playernumber in range(5):
    account_id = match_data['participantIdentities'][playernumber]['player']['accountId']
    player_account_id_collection.append(account_id)

# Get the match history of each player
player_matchlist_collection = []
for player_account_id in player_account_id_collection:
    matchlist = watcher.match.matchlist_by_account(my_region,player_account_id)
    just_the_match_ids = []
    for match in matchlist['matches']:
        just_the_match_ids.append(match['gameId'])
    # We only want the games that occurred prior to the game being analyzed
    largest_match_num = just_the_match_ids[0]
    while int(largest_match_num) >= int(matchidnumber):
        just_the_match_ids.pop(0)
        largest_match_num = just_the_match_ids[0]
    just_the_match_ids = just_the_match_ids[:18]
    player_matchlist_collection.append(just_the_match_ids)
#pprint.pprint(player_matchlist_collection)

# Get stats from the list of matches compiled above
player_account_id_to_analyze = player_account_id_collection[0]
match_to_analyze = player_matchlist_collection[0][0]
match_data_2 = watcher.match.by_id(my_region,match_to_analyze)
# Get the participant ID for our player account number
for player_list in range(10):
    if int(player_account_id_to_analyze) == int(match_data_2['participantIdentities'][player_list]['player']['accountId']):
        participant_ID = match_data_2['participantIdentities'][player_list]['participantId']
        print(participant_ID)
# Collect stats for our player from that game, and add them to the a list
player_game_stats = match_data_2['participants'][int(participant_ID)-1]['stats']
sorted_game_stats = []
keylist = list(player_game_stats.keys())
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

pprint.pprint(final_stats_list)
pprint.pprint(match_data_2)
