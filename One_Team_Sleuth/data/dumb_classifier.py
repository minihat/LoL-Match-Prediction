import csv

csvfile = open('test2.csv','r')
reader = csv.reader(csvfile)

csvfile = open('analyzed_test2.csv','a')
bibowriter = csv.writer(csvfile,delimiter=',')

total_correct = 0
total_analyzed = 0
for game_data in reader:
    try:
        write_stats = []
        print(game_data)
        overall_rating = 0.0
        for i in range(7):
            j = i + 2
            l = j + 9
            #print(j,l)
            stat_ratio = float(game_data[j])/float(game_data[l])
            overall_rating += (stat_ratio - 1)
            print(stat_ratio - 1)
            write_stats.append(stat_ratio - 1)

        assists_ratio = float(game_data[0])/float(game_data[9])
        print(assists_ratio - 1)
        overall_rating += (assists_ratio - 1)
        write_stats.append(assists_ratio - 1)
        deaths_ratio = float(game_data[1])/float(game_data[10])
        print(-(deaths_ratio - 1))
        overall_rating += -(deaths_ratio - 1)
        write_stats.append(-(deaths_ratio - 1))

        print(overall_rating)

        if overall_rating < 0:
            prediction = 0

        if overall_rating >= 0:
            prediction = 1
        if int(prediction) == int(game_data[18]):
            total_correct += 1
        total_analyzed += 1

        write_stats.append(int(game_data[18]))
        bibowriter.writerow(write_stats)
        csvfile.flush()

        print("Prediction: " + str(prediction))
        print("Actual: " + str(game_data[18]))
        print("Correct Ratio: " + str(float(total_correct)/float(total_analyzed)))
        #input("Press enter to continue...")
    except:
        print("Skipped a Row.")
