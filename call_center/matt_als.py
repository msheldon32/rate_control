import csv

import heapq

state_dwell = [0]
state_arrivals = [0]
state_departures = [0]

entry_times = []
departure_times = []

with open("preprocessed.csv") as f:
    reader = csv.DictReader(f)

    cur_state = 0
    cur_date = ""

    t = 0

    departures = []

    epoch = 10*3600

    time_offset = 0

    for row in reader:
        new_offset = False
        if row["date"] != cur_date:
            cur_date = row["date"]
            time_offset += 3*3600
            new_offset = True

        entry_t = int(row["q_start"])-epoch+time_offset

        if row["outcome"] == "HANG":
            departure_t = int(row["q_exit"])-epoch+time_offset
        else:
            departure_t = int(row["ser_exit"])-epoch+time_offset
        
        while departures and departures[0] < entry_t:
            state_departures[cur_state] += 1
            state_dwell[cur_state] += departures[0] - t
            t = departures[0]
            print(f"t: {t}")
            heapq.heappop(departures)
            cur_state -= 1

        state_arrivals[cur_state] += 1
        state_dwell[cur_state] += entry_t - t
        t = entry_t
        print(f"t: {t}")
        cur_state += 1
        
        if cur_state >= len(state_dwell):
            state_dwell.append(0)
            state_arrivals.append(0)
            state_departures.append(0)

        heapq.heappush(departures, departure_t)

print(state_dwell)
print([x/y for x, y in zip(state_arrivals, state_dwell)])
print([x/y for x, y in zip(state_departures, state_dwell)])


print(state_arrivals)
print(state_departures)
