from riotwatcher import RiotWatcher
import pprint
import sys

watcher = RiotWatcher('RGAPI-a6d0d16d-146a-48fb-b402-660210dca566')

my_region = 'na1'

me = watcher.summoner.by_name(my_region, 'Lahourla')
player_account_id = me['accountId']
matchlist = watcher.match.matchlist_by_account(my_region,player_account_id)
print(matchlist)
